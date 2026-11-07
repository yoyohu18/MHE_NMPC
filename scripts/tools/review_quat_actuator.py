#!/usr/bin/env python3
"""Thesis review 2 (2026-11-06): quaternion RK4 norm drift over the NMPC horizon and
per-rotor feasibility of the NMPC (T, tau) box. Usage: python3 review_quat_actuator.py"""
import numpy as np, glob, csv
# ---- quaternion RK4 drift, NMPC settings: ERK4, 1 step, dt=0.05, N=20
def Xi(q):
    qw,qx,qy,qz=q
    return np.array([[-qx,-qy,-qz],[qw,-qz,qy],[qz,qw,-qx],[-qy,qx,qw]])
def rk4(q,w,h):
    f=lambda q:0.5*Xi(q)@w
    k1=f(q);k2=f(q+h/2*k1);k3=f(q+h/2*k2);k4=f(q+h*k3)
    return q+h/6*(k1+2*k2+2*k3+k4)
h=0.05;N=20
for wn in [1,2,3,5,8]:
    q=np.array([1.,0,0,0]);w=np.array([wn,0,0.])
    for _ in range(N): q=rk4(q,w,h)
    print(f'|w|={wn} rad/s: | |q_N|-1 | = {abs(np.linalg.norm(q)-1):.2e}')
# logged body rates, all 96 flights: horizon drift using the measured rate sequence
D='/home/clear/ros2_ws_HJH/nmpc_test_results/mhe_variant_ab_20261105_181925/main/flights'
worst=[];wmax=0
for f in sorted(glob.glob(D+'/*/resid/resid_odom_*.csv')):
    a=np.genfromtxt(f,delimiter=',',names=True)
    t=a['t_recv_ns']*1e-9; W=np.column_stack([a['wb_x'],a['wb_y'],a['wb_z']])
    tg=np.arange(t[0],t[-1],h); Wg=np.column_stack([np.interp(tg,t,W[:,i]) for i in range(3)])
    wmax=max(wmax,np.abs(Wg).max())
    dmax=0
    for s in range(0,len(tg)-N,20):
        q=np.array([1.,0,0,0])
        for j in range(N): q=rk4(q,Wg[s+j],h)
        dmax=max(dmax,abs(np.linalg.norm(q)-1))
    worst.append(dmax)
print('flights',len(worst),'max horizon drift over all flights %.2e, median of per-flight max %.2e, max |w_i| %.2f'%(max(worst),np.median(worst),wmax))
# ---- per-rotor feasibility of the NMPC input box
kf=8.54858e-06; km=0.016
X=np.array([0.174,-0.174,0.174,-0.174]); Y=np.array([-0.174,0.174,0.174,-0.174]); D_=np.array([1.,1,-1,-1])
M=np.vstack([np.ones(4), Y, -X, km*D_])        # [T,tx,ty,tz] = M F  (manifest tau_phys_def)
Mi=np.linalg.inv(M)
Fmin=kf*(150+850*0.05)**2; Fmax=kf*(150+850*0.95)**2
Tmin,Tmax=4*Fmin,4*Fmax
print('per-rotor F in [%.3f, %.3f] N; box T in [%.2f, %.2f] N'%(Fmin,Fmax,Tmin,Tmax))
def feas(T,t): F=Mi@np.r_[T,t]; return np.all(F>=Fmin-1e-9) and np.all(F<=Fmax+1e-9)
for T in [20.25, 21.7, 23.19, 26, 28]:
    # largest scalar s such that each single-axis torque at s*limit is feasible
    out=[]
    for ax,lim in [(0,0.5),(1,0.5),(2,0.2)]:
        lo,hi=0,1
        for _ in range(40):
            mid=(lo+hi)/2; tt=np.zeros(3); tt[ax]=mid*lim
            ok=feas(T,tt) and feas(T,-tt); lo,hi=(mid,hi) if ok else (lo,mid)
        out.append(lo*lim)
    corner=all(feas(T,np.array([sx*0.5,sy*0.5,sz*0.2])) for sx in(-1,1) for sy in(-1,1) for sz in(-1,1))
    print(f'T={T:5.2f} N: max feasible |tau_x|={out[0]:.3f} |tau_y|={out[1]:.3f} |tau_z|={out[2]:.3f} Nm; full box corners feasible={corner}')
# fraction of the box volume that is infeasible
rng=np.random.default_rng(0); S=rng.uniform([Tmin,-.5,-.5,-.2],[Tmax,.5,.5,.2],(200000,4))
inf=np.array([not feas(s[0],s[1:]) for s in S]); print('infeasible share of the full input box: %.3f'%inf.mean())
S2=S[(S[:,0]>18)&(S[:,0]<26)]; inf2=np.array([not feas(s[0],s[1:]) for s in S2]); print('infeasible share for T in [18,26] N: %.3f'%inf2.mean())
