// Gazebo (gz-sim8) WORLD system plugin: a "magnetic gripper" proxy.
//
// On an attach command it creates a *fixed* DetachableJoint between an
// arbitrary parent link and an arbitrary child link, both named at runtime
// in the message payload; on a detach command it removes that joint entity.
// Detach-then-attach is supported (repeatable). It does NOT model finger
// contact dynamics on purpose - the research question is how the flight
// controller responds to a step change in payload mass / CoM / inertia, not
// grasp mechanics, so a rigid fixed joint standing in for the gripper is the
// minimal faithful model.
//
// Why a custom WORLD plugin instead of gz-sim's built-in DetachableJoint
// system (which in gz-sim8 *does* support attach/detach topics and repeat
// attach): the built-in bakes <child_model>/<child_link> into the SDF at
// load time and is a model-level plugin whose parent is always its host
// model. It therefore cannot pick an arbitrary target at runtime from a
// candidate list, and - the decisive point for this rig - it creates the
// joint at Configure time. A joint that exists from t=0 welds the vehicle
// to a grounded/distant body before takeoff and sends the controller into a
// violent, unrecoverable oscillation (this exact failure was hit repeatedly
// on this stack with the built-in approach). This plugin creates the joint
// ONLY when an attach command arrives - i.e. only once the proximity node
// has confirmed the vehicle is hovering right over the target with near-zero
// relative velocity - so the constraint locks a small relative pose with no
// velocity mismatch and no impulse.
#ifndef MAGNETIC_GRIPPER_HH_
#define MAGNETIC_GRIPPER_HH_

#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <gz/sim/System.hh>
#include <gz/sim/Entity.hh>
#include <gz/transport/Node.hh>
#include <gz/msgs/stringmsg.pb.h>

namespace magnetic_gripper
{

// One pending command parsed off the transport thread, applied later on the
// simulation thread in PreUpdate.
struct GripperRequest
{
  bool attach{true};               // true = attach, false = detach
  std::string parentModel;
  std::string parentLink;
  std::string childModel;
  std::string childLink;
};

class MagneticGripper:
    public gz::sim::System,
    public gz::sim::ISystemConfigure,
    public gz::sim::ISystemPreUpdate
{
  public: MagneticGripper() = default;
  public: ~MagneticGripper() override = default;

  public: void Configure(
      const gz::sim::Entity &_entity,
      const std::shared_ptr<const sdf::Element> &_sdf,
      gz::sim::EntityComponentManager &_ecm,
      gz::sim::EventManager &_eventMgr) override;

  public: void PreUpdate(
      const gz::sim::UpdateInfo &_info,
      gz::sim::EntityComponentManager &_ecm) override;

  // gz-transport receive-thread callbacks. They only parse + enqueue; they
  // never touch the ECM (no thread-safety guarantee). The actual entity
  // create/remove happens in PreUpdate on the simulation thread.
  private: void OnAttach(const gz::msgs::StringMsg &_msg);
  private: void OnDetach(const gz::msgs::StringMsg &_msg);

  // Parse "parentModel parentLink childModel childLink" (whitespace
  // separated). Returns false on a malformed payload.
  private: bool ParsePayload(const std::string &_data, GripperRequest &_req);

  private: void DoAttach(const GripperRequest &_req,
      const gz::sim::UpdateInfo &_info,
      gz::sim::EntityComponentManager &_ecm);
  private: void DoDetach(const GripperRequest &_req,
      const gz::sim::UpdateInfo &_info,
      gz::sim::EntityComponentManager &_ecm);

  // Publish a one-line, sim-time-stamped status string so downstream tools
  // (e.g. an estimator-reset experiment) can key off the exact attach/detach
  // instants.
  private: void PublishState(const std::string &_line);

  private: gz::sim::Entity worldEntity{gz::sim::kNullEntity};
  private: gz::transport::Node node;
  private: gz::transport::Node::Publisher statePub;

  private: std::string attachTopic{"/gripper/attach"};
  private: std::string detachTopic{"/gripper/detach"};
  private: std::string stateTopic{"/gripper/state"};

  // Pending requests, guarded because they are filled from transport
  // threads and drained on the sim thread.
  private: std::mutex reqMutex;
  private: std::vector<GripperRequest> pending;

  // Active joints, keyed by "parentModel/parentLink|childModel/childLink",
  // so a repeated attach is a no-op and detach can find the entity to drop.
  private: std::map<std::string, gz::sim::Entity> joints;
};

}  // namespace magnetic_gripper

#endif
