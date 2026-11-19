"""Bounded synthetic validation of the existing readout-cost theory.

Protocol fixed before execution: seed 20261110; 8 networks with 2/4/6/8
independent recurrent pathways; windows .35/1.6; budgets .25/1/4 times
the unit-gain cost. No filtering on effect size. All optimizer outcomes saved.
"""
from pathlib import Path
import json
import warnings
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import block_diag, expm, solve_continuous_lyapunov
from scipy.optimize import brentq, minimize

HERE = Path(__file__).resolve().parent
SEED = 20261110


def covariance(j, q):
    assert np.max(np.linalg.eigvals(j).real) < 0
    s = solve_continuous_lyapunov(j, -q)
    s = (s+s.T)/2
    assert np.linalg.norm(j@s+s@j.T+q) < 1e-9*(1+np.linalg.norm(q))
    return s


def moments(j, q, cue, c, t):
    s = covariance(j, q)
    e = expm(j*t)
    h = c @ np.linalg.solve(j, (e-np.eye(len(j))) @ cue)/t
    integ = np.linalg.solve(j, np.linalg.solve(j, e-np.eye(len(j))-j*t))
    v = 2*c @ integ @ s @ c/t**2
    assert np.isfinite([h, v]).all() and v > 0
    return float(h), float(v), s


def generator(route, m):
    a, d, b = route['A'], route['D'], route['B']
    return np.block([[d, m*b], [np.zeros((len(a), len(d))), a]])


def coefficients(route, t):
    """Unit-gain, separate-noise quadrature; no fit of gain/performance curves."""
    ns, no = len(route['A']), len(route['D'])
    j = generator(route, 1.)
    qs = block_diag(np.zeros((no,no)), route['Qs'])
    qo = block_diag(route['Qo'], np.zeros((ns,ns)))
    ss, so = covariance(j, qs), covariance(generator(route, 0.), qo)
    kappa = np.trace(ss[:no,:no])
    z, w = leggauss(48)
    ts, ws = (z+1)*t/2, w*t/2
    h = sum(ww*(route['c']@expm(j*u)@route['cue']) for u,ww in zip(ts,ws))/t
    local = 2*sum(ww*(t-u)*(route['c']@expm(generator(route,0.)*u)@so@route['c'])
                  for u,ww in zip(ts,ws))/t**2
    prop = 2*sum(ww*(t-u)*(route['c']@expm(j*u)@ss@route['c'])
                 for u,ww in zip(ts,ws))/t**2
    assert min(kappa, local, prop, h*h)>0
    return {'h':float(h), 'L':float(local), 'S':float(prop),
            'kappa':float(kappa), 'F':float(4*h*h/prop),
            'price':float(kappa*local/prop)}


def allocation(b, f, c):
    high = np.max(f/c)
    def x(lam): return np.maximum(np.sqrt(f*c/lam)-c,0.)
    lam=brentq(lambda z:x(z).sum()-b, high*1e-16, high, xtol=1e-15,rtol=1e-13)
    return x(lam)


def fair_allocation(b, f, c):
    y=brentq(lambda y:np.sum(c*y/(f-y))-b,0.,np.min(f)*(1-1e-13),xtol=1e-14)
    return c*y/(f-y)


def new_route(rng, case, i):
    ns, no = 2+(case+i)%3, 2+(2*case+i)%3
    def internal(n, scale):
        x=rng.uniform(.02,.20,(n,n)); np.fill_diagonal(x,0.)
        if case%2: x *= rng.choice([-1.,1.],x.shape)
        return scale*(x-np.diag(np.abs(x).sum(axis=1)+rng.uniform(.35,1.1,n)))
    a=internal(ns,float(rng.uniform(.5,1.6)))
    d=internal(no,float(rng.uniform(.5,1.6)))
    b=rng.uniform(.1,1.,(no,ns));b/=np.linalg.norm(b)
    qs=np.diag(rng.uniform(.35,1.7,ns));qo=np.diag(rng.uniform(.35,1.7,no))
    return {'A':a,'D':d,'B':b,'Qs':qs,'Qo':qo,
            'Q':block_diag(qo,qs),'cue':np.r_[np.zeros(no),np.ones(ns)/np.sqrt(ns)],
            'c':np.r_[np.ones(no)/np.sqrt(no),np.zeros(ns)]}


def recurrent_prediction():
    rng=np.random.default_rng(SEED)
    networks=[]; rows=[]
    for case in range(8):
        n=(2,4,6,8)[case%4]
        routes=[new_route(rng,case,i) for i in range(n)]
        coefs={t:[coefficients(r,t) for r in routes] for t in (.35,1.6)}
        kap=np.array([c['kappa'] for c in coefs[.35]])
        np.testing.assert_allclose(kap,[c['kappa'] for c in coefs[1.6]],rtol=1e-12)
        b0=float(kap.sum())
        base=[np.trace(covariance(generator(r,0.),r['Q'])) for r in routes]
        networks.append({'case':case,'pathways':n,'unit_gain_cost':b0,
            'routes':[{k:v.tolist() for k,v in r.items()} for r in routes],
            'coefficients':{str(t):coefs[t] for t in coefs}})
        for t in (.35,1.6):
            cs=coefs[t]; f=np.array([z['F'] for z in cs]); c=np.array([z['price'] for z in cs])
            for mult in (.25,1.,4.):
                budget=mult*b0
                pred=allocation(budget,f,c)/budget
                pfair=fair_allocation(budget,f,c)/budget
                cache={}
                def direct(z):
                    key=tuple(np.round(np.maximum(z,0.),14))
                    if key not in cache:
                        scores=[]; costs=[]
                        for i,r in enumerate(routes):
                            m=np.sqrt(budget*max(z[i],0.)/kap[i])
                            h,v,s=moments(generator(r,m),r['Q'],r['cue'],r['c'],t)
                            scores.append(4*h*h/v);costs.append(np.trace(s)-base[i])
                        cache[key]=(np.array(scores),float(sum(costs)))
                    return cache[key]
                scale=float(f.sum())
                starts=[np.ones(n)/n,rng.dirichlet(np.ones(n)),np.eye(n)[0]]
                attempts=[]
                for start in starts:
                    opt=minimize(lambda z:-direct(z)[0].sum()/scale,start,method='SLSQP',
                        bounds=[(0.,1.)]*n,
                        constraints={'type':'eq','fun':lambda z:z.sum()-1.},
                        options={'ftol':2e-13,'maxiter':400})
                    attempts.append({'success':bool(opt.success),'message':str(opt.message),
                        'shares':opt.x.tolist(),'utility':float(direct(opt.x)[0].sum())})
                usable=[z for z in attempts if z['success']]
                assert usable, attempts
                best=max(usable,key=lambda z:z['utility']); dx=np.array(best['shares'])
                u_pred=float(direct(pred)[0].sum()); u_equal=float(direct(np.ones(n)/n)[0].sum())
                u_formula=float(np.sum(f*budget*pred/(budget*pred+c)))
                assert abs(u_pred-u_formula)<1e-9*max(1.,u_pred)
                gap=abs(u_pred-best['utility'])/max(u_pred,1e-14)
                assert gap<3e-7,(case,t,mult,gap)
                assert abs(direct(pred)[1]/budget-1.)<2e-10
                # Epigraph optimization uses full-state scores, not the closed readout law.
                start=np.r_[np.ones(n)/n,.5*min(direct(np.ones(n)/n)[0])/max(f)]
                fo=minimize(lambda z:-z[-1],start,method='SLSQP',
                    bounds=[(0.,1.)]*n+[(0.,1.)],constraints=[
                    {'type':'eq','fun':lambda z:z[:-1].sum()-1.},
                    {'type':'ineq','fun':lambda z:direct(z[:-1])[0]/max(f)-z[-1]}],
                    options={'ftol':2e-13,'maxiter':600})
                fair_actual=float(min(direct(pfair)[0]));fair_direct=float(min(direct(fo.x[:-1])[0]))
                assert fo.success,(case,t,mult,fo.message)
                fair_gap=abs(fair_actual-fair_direct)/max(fair_actual,1e-14)
                assert fair_gap<5e-7,(case,t,mult,fair_gap)
                rows.append({'network':case,'n':n,'T':t,'budget_multiplier':mult,'budget':budget,
                    'predicted_shares':pred.tolist(),'direct_shares':dx.tolist(),
                    'fair_predicted_shares':pfair.tolist(),'fair_direct_shares':fo.x[:-1].tolist(),
                    'joint_relative_objective_error':gap,'fair_relative_objective_error':fair_gap,
                    'share_max_abs_error':float(max(abs(pred-dx))),
                    'joint_predicted':u_pred,'joint_direct':best['utility'],'joint_equal':u_equal,
                    'gain_over_equal_percent':100*(u_pred/u_equal-1.),
                    'cost_relative_error':abs(direct(pred)[1]/budget-1.),'attempts':attempts,
                    'fair_solver_success':bool(fo.success)})
        print('Completed recurrent network',case,'with',n,'pathways',flush=True)
    shifts=[]
    for k in range(8):
        for mult in (.25,1.,4.):
            a,b=[z for z in rows if z['network']==k and z['budget_multiplier']==mult]
            shifts.append({'network':k,'budget_multiplier':mult,
                'joint_share_total_variation':float(sum(abs(np.array(a['predicted_shares'])-b['predicted_shares']))/2),
                'fair_share_total_variation':float(sum(abs(np.array(a['fair_predicted_shares'])-b['fair_predicted_shares']))/2)})
    gains=[z['gain_over_equal_percent'] for z in rows]
    return {'seed':SEED,'networks':networks,'rows':rows,'shifts':shifts,'summary':{
        'networks':8,'pathways':sum(z['pathways'] for z in networks),'settings':len(rows),
        'max_joint_relative_error':max(z['joint_relative_objective_error'] for z in rows),
        'max_fair_relative_error':max(z['fair_relative_objective_error'] for z in rows),
        'max_share_absolute_error':max(z['share_max_abs_error'] for z in rows),
        'gain_over_equal_percent_range':[min(gains),max(gains)],
        'median_gain_over_equal_percent':float(np.median(gains)),
        'gain_median_by_window':{str(t):float(np.median([z['gain_over_equal_percent'] for z in rows if z['T']==t])) for t in (.35,1.6)},
        'median_joint_share_shift':float(np.median([z['joint_share_total_variation'] for z in shifts])),
        'max_cost_relative_error':max(z['cost_relative_error'] for z in rows),
        'failed_joint_optimizer_starts':sum(not a['success'] for z in rows for a in z['attempts'])}}


def local(a,t):
    return (a*t+np.expm1(-a*t))/(a**3*t*t)


def pair_quantities(a,b,t):
    h=((-np.expm1(-b*t))/b-(-np.expm1(-a*t))/a)/((a-b)*t*np.sqrt(2))
    s=(local(a,t)-local(b,t))/(b*b-a*a)
    return h,s


def pair_scores(a,b,ms,mf,t,source=(1.,1.),local_drive=(1.,1.)):
    h,s=pair_quantities(a,b,t)
    return np.array([2*h*ms/np.sqrt(local_drive[0]*local(a,t)+ms*ms*s*source[0]),
                     2*h*mf/np.sqrt(local_drive[1]*local(b,t)+mf*mf*s*source[1])])


def reversal_region():
    rates=np.linspace(1.05,8.,141);gammas=np.linspace(1.01,8.,161)
    contrast=np.full((len(gammas),len(rates)),np.nan)
    tested=0; finite=0; pointwise=0
    for j,r in enumerate(rates):
        a,b=1/np.sqrt(r),np.sqrt(r);kap=1/(2*a*b*(a+b))
        for i,g in enumerate(gammas):
            mf=np.sqrt(1.25712/(kap*(1+g*g)));ms=g*mf
            lo=pair_scores(a,b,ms,mf,.5);hi=pair_scores(a,b,ms,mf,3.)
            short=lo[0]/lo[1];long=hi[0]/hi[1]
            prediction=(g*g>local(a,.5)/local(b,.5) and g*g<local(a,3.)/local(b,3.))
            actual=short>1 and long<1
            assert prediction==actual
            tested+=1;pointwise+=int(actual)
            if actual:
                contrast[i,j]=100*min(short-1,1-long);finite+=1
    # Independent full-state checks on a prespecified coarse subset.
    for r in (1.2,2.,4.,7.):
        a,b=1/np.sqrt(r),np.sqrt(r)
        for g in (1.1,1.7,2.6,4.5):
            kap=1/(2*a*b*(a+b));mf=np.sqrt(1.25712/(kap*(1+g*g)));ms=g*mf
            for t in (.5,3.):
                expected=pair_scores(a,b,ms,mf,t)
                actual=[]
                for out,src,m in [(a,b,ms),(b,a,mf)]:
                    j=np.array([[-out,m],[0.,-src]])
                    h,v,_=moments(j,np.eye(2),np.array([0.,1/np.sqrt(2)]),np.array([1.,0.]),t)
                    actual.append(2*h/np.sqrt(v))
                np.testing.assert_allclose(actual,expected,rtol=3e-11)
    times=np.geomspace(.025,12.,240)
    ablations={}
    for name,src,loc in [('Baseline',(1.,1.),(1.,1.)),('No local noise',(1.,1.),(0.,0.)),('No source noise',(0.,0.),(1.,1.))]:
        vals=np.array([pair_scores(.5,2.,2.34,.9,t,src,loc) for t in times])
        ablations[name]={'slow':vals[:,0].tolist(),'fast':vals[:,1].tolist(),'ratio':(vals[:,0]/vals[:,1]).tolist()}
    # Deliberate failure of the shared-source-noise/price assumptions at equal budget.
    # Source intensity in slow route is fourfold; actual allocated x ratio stays 6.76.
    kap=.2;ratio=6.76;xf=1.25712/(1+ratio);xs=1.25712-xf
    ms,mf=np.sqrt(xs/(4*kap)),np.sqrt(xf/kap)
    mismatch=np.array([pair_scores(.5,2.,ms,mf,t,source=(4.,1.)) for t in times])
    assert (mismatch[:,0]/mismatch[:,1]<1).all()
    return {'rates':rates.tolist(),'gammas':gammas.tolist(),
        'strength_percent':[[None if not np.isfinite(v) else float(v) for v in row] for row in contrast],
        'grid_count':tested,'fixed_window_reversals':finite,'full_state_checks':32,
        'window_durations':[.5,3.],'fixed_excess_budget':1.25712,'times':times.tolist(),
        'ablations':ablations,'mismatch':{'source_intensities':[4.,1.],'gains':[float(ms),float(mf)],
            'fixed_budget_ratio':ratio,'ratio':(mismatch[:,0]/mismatch[:,1]).tolist(),
            'endpoint_condition_is_inapplicable':True,'crossing_in_sampled_range':False}}


def main():
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',category=RuntimeWarning,message='.*encountered in matmul')
        record={'protocol':{'seed':SEED,'networks':8,'pathways_per_network':[2,4,6,8,2,4,6,8],
                           'windows':[.35,1.6],'budget_multipliers':[.25,1.,4.],
                           'fixed_grid_before_execution':True,'biological_data':False},
                'recurrent':recurrent_prediction(),'region':reversal_region(),'passed':True}
    (HERE/'strengthening_results.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(record['recurrent']['summary'],indent=2))
    print('All bounded validation checks PASS',flush=True)


if __name__=='__main__': main()
