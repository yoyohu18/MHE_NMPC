#include "MagneticGripper.hh"

#include <chrono>
#include <sstream>

#include <gz/plugin/Register.hh>
#include <gz/common/Console.hh>
#include <gz/common/Profiler.hh>
#include <gz/sim/World.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/components/DetachableJoint.hh>
#include <gz/sim/components/ParentEntity.hh>

using namespace magnetic_gripper;

namespace
{
  // Key identifying one (parent link, child link) weld.
  std::string JointKey(const GripperRequest &_req)
  {
    return _req.parentModel + "/" + _req.parentLink + "|" +
           _req.childModel + "/" + _req.childLink;
  }

  double SimSeconds(const gz::sim::UpdateInfo &_info)
  {
    return std::chrono::duration<double>(_info.simTime).count();
  }
}

//////////////////////////////////////////////////
void MagneticGripper::Configure(
    const gz::sim::Entity &_entity,
    const std::shared_ptr<const sdf::Element> &_sdf,
    gz::sim::EntityComponentManager &_ecm,
    gz::sim::EventManager & /*_eventMgr*/)
{
  this->worldEntity = _entity;
  gz::sim::World world(_entity);
  if (!world.Valid(_ecm))
  {
    gzerr << "[MagneticGripper] must be attached to a <world>; got a "
          << "non-world entity. Plugin disabled.\n";
    this->worldEntity = gz::sim::kNullEntity;
    return;
  }

  if (_sdf->HasElement("attach_topic"))
    this->attachTopic = _sdf->Get<std::string>("attach_topic");
  if (_sdf->HasElement("detach_topic"))
    this->detachTopic = _sdf->Get<std::string>("detach_topic");
  if (_sdf->HasElement("state_topic"))
    this->stateTopic = _sdf->Get<std::string>("state_topic");

  this->node.Subscribe(this->attachTopic, &MagneticGripper::OnAttach, this);
  this->node.Subscribe(this->detachTopic, &MagneticGripper::OnDetach, this);
  this->statePub =
      this->node.Advertise<gz::msgs::StringMsg>(this->stateTopic);

  gzmsg << "[MagneticGripper] ready. attach=" << this->attachTopic
        << " detach=" << this->detachTopic
        << " state=" << this->stateTopic << "\n";
}

//////////////////////////////////////////////////
bool MagneticGripper::ParsePayload(
    const std::string &_data, GripperRequest &_req)
{
  std::istringstream iss(_data);
  if (iss >> _req.parentModel >> _req.parentLink
          >> _req.childModel >> _req.childLink)
    return true;
  return false;
}

//////////////////////////////////////////////////
void MagneticGripper::OnAttach(const gz::msgs::StringMsg &_msg)
{
  GripperRequest req;
  req.attach = true;
  if (!this->ParsePayload(_msg.data(), req))
  {
    gzwarn << "[MagneticGripper] malformed attach payload: '"
           << _msg.data() << "' (want 'parentModel parentLink "
           << "childModel childLink')\n";
    return;
  }
  std::lock_guard<std::mutex> lock(this->reqMutex);
  this->pending.push_back(req);
}

//////////////////////////////////////////////////
void MagneticGripper::OnDetach(const gz::msgs::StringMsg &_msg)
{
  GripperRequest req;
  req.attach = false;
  if (!this->ParsePayload(_msg.data(), req))
  {
    gzwarn << "[MagneticGripper] malformed detach payload: '"
           << _msg.data() << "'\n";
    return;
  }
  std::lock_guard<std::mutex> lock(this->reqMutex);
  this->pending.push_back(req);
}

//////////////////////////////////////////////////
void MagneticGripper::PreUpdate(
    const gz::sim::UpdateInfo &_info,
    gz::sim::EntityComponentManager &_ecm)
{
  GZ_PROFILE("MagneticGripper::PreUpdate");
  if (this->worldEntity == gz::sim::kNullEntity || _info.paused)
    return;

  // Drain pending requests under the lock, then act without holding it.
  std::vector<GripperRequest> todo;
  {
    std::lock_guard<std::mutex> lock(this->reqMutex);
    if (this->pending.empty())
      return;
    todo.swap(this->pending);
  }

  for (const auto &req : todo)
  {
    if (req.attach)
      this->DoAttach(req, _info, _ecm);
    else
      this->DoDetach(req, _info, _ecm);
  }
}

//////////////////////////////////////////////////
void MagneticGripper::DoAttach(
    const GripperRequest &_req,
    const gz::sim::UpdateInfo &_info,
    gz::sim::EntityComponentManager &_ecm)
{
  const std::string key = JointKey(_req);
  if (this->joints.count(key) != 0)
  {
    gzmsg << "[MagneticGripper] already attached: " << key << " (ignored)\n";
    return;
  }

  gz::sim::World world(this->worldEntity);

  gz::sim::Entity parentModelE =
      world.ModelByName(_ecm, _req.parentModel);
  if (parentModelE == gz::sim::kNullEntity)
  {
    gzwarn << "[MagneticGripper] attach failed: no model named '"
           << _req.parentModel << "'\n";
    return;
  }
  gz::sim::Entity childModelE = world.ModelByName(_ecm, _req.childModel);
  if (childModelE == gz::sim::kNullEntity)
  {
    gzwarn << "[MagneticGripper] attach failed: no model named '"
           << _req.childModel << "'\n";
    return;
  }
  if (parentModelE == childModelE)
  {
    // DetachableJoint connecting a model to itself is silently a no-op in
    // the physics system - reject it loudly instead of looking attached.
    gzwarn << "[MagneticGripper] attach failed: parent and child are the "
           << "same model '" << _req.parentModel << "'. DetachableJoint "
           << "only works across two distinct models.\n";
    return;
  }

  gz::sim::Model parentModel(parentModelE);
  gz::sim::Model childModel(childModelE);
  gz::sim::Entity parentLinkE =
      parentModel.LinkByName(_ecm, _req.parentLink);
  gz::sim::Entity childLinkE = childModel.LinkByName(_ecm, _req.childLink);
  if (parentLinkE == gz::sim::kNullEntity)
  {
    gzwarn << "[MagneticGripper] attach failed: model '" << _req.parentModel
           << "' has no link '" << _req.parentLink << "'\n";
    return;
  }
  if (childLinkE == gz::sim::kNullEntity)
  {
    gzwarn << "[MagneticGripper] attach failed: model '" << _req.childModel
           << "' has no link '" << _req.childLink << "'\n";
    return;
  }

  // Create the joint entity and tag it with a DetachableJoint component;
  // the physics system turns that into an actual fixed constraint locking
  // the current relative pose of the two links. The joint entity is
  // parented to the child model so it lives/dies with a sensible owner.
  gz::sim::Entity jointE = _ecm.CreateEntity();
  _ecm.CreateComponent(jointE,
      gz::sim::components::DetachableJoint(
          {parentLinkE, childLinkE, "fixed"}));
  _ecm.CreateComponent(jointE,
      gz::sim::components::ParentEntity(childModelE));

  this->joints[key] = jointE;

  const double t = SimSeconds(_info);
  gzmsg << "[MagneticGripper] ATTACHED " << key << " jointEntity=" << jointE
        << " sim_time=" << t << "s\n";

  std::ostringstream line;
  line << "ATTACHED " << key << " sim_time=" << t
       << " entity=" << jointE;
  this->PublishState(line.str());
}

//////////////////////////////////////////////////
void MagneticGripper::DoDetach(
    const GripperRequest &_req,
    const gz::sim::UpdateInfo &_info,
    gz::sim::EntityComponentManager &_ecm)
{
  const std::string key = JointKey(_req);
  auto it = this->joints.find(key);
  if (it == this->joints.end())
  {
    gzmsg << "[MagneticGripper] already detached / never attached: " << key
          << " (ignored)\n";
    return;
  }

  const gz::sim::Entity jointE = it->second;
  _ecm.RequestRemoveEntity(jointE);
  this->joints.erase(it);

  const double t = SimSeconds(_info);
  gzmsg << "[MagneticGripper] DETACHED " << key << " jointEntity=" << jointE
        << " sim_time=" << t << "s\n";

  std::ostringstream line;
  line << "DETACHED " << key << " sim_time=" << t
       << " entity=" << jointE;
  this->PublishState(line.str());
}

//////////////////////////////////////////////////
void MagneticGripper::PublishState(const std::string &_line)
{
  if (!this->statePub)
    return;
  gz::msgs::StringMsg msg;
  msg.set_data(_line);
  this->statePub.Publish(msg);
}

GZ_ADD_PLUGIN(
    magnetic_gripper::MagneticGripper,
    gz::sim::System,
    magnetic_gripper::MagneticGripper::ISystemConfigure,
    magnetic_gripper::MagneticGripper::ISystemPreUpdate)

GZ_ADD_PLUGIN_ALIAS(magnetic_gripper::MagneticGripper,
    "magnetic_gripper::MagneticGripper")
