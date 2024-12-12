# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
import numpy as np 
"""
    Author: S. Schwalbe
    Date: 12.12.2024
"""

class Parameters: 
    """
        Parameters class 

        Holds parameters of exchange-correlation 
        functionals. 
        Saving same space through attribute access. 
    """
    def __init__(self,params): 
        for key,val in params.items(): 
            setattr(self,key,val) 

def get_b5(omega,b3):
    """
        Get b5. 
    """
    b5 = b3*np.sqrt(3/2) *omega* (4/(9*np.pi))**(-1/3)
    return b5 

def get_gdsmfb_parameters(): 
    """
        Get the GDSMFB parameters. 
    """
    # zeta = 0
    p_zeta0 = {}
    p_zeta0["omega"] = 1
    p_zeta0["b1"] = 0.3436902
    p_zeta0["b2"] = 7.82159531356
    p_zeta0["b3"] = 0.300483986662
    p_zeta0["b4"] = 15.8443467125
    p_zeta0["b5"] = get_b5(p_zeta0["omega"],p_zeta0["b3"])
    p_zeta0["c1"] = 0.8759442
    p_zeta0["c2"] =-0.230130843551
    p_zeta0["d1"] = 0.72700876
    p_zeta0["d2"] = 2.38264734144
    p_zeta0["d3"] = 0.30221237251
    p_zeta0["d4"] = 4.39347718395
    p_zeta0["d5"] = 0.729951339845
    p_zeta0["e1"] = 0.25388214
    p_zeta0["e2"] = 0.815795138599
    p_zeta0["e3"] = 0.0646844410481
    p_zeta0["e4"] = 15.0984620477
    p_zeta0["e5"] = 0.230761357474
    p0 = Parameters(p_zeta0) 

    # zeta = 1
    p_zeta1 = {}
    p_zeta1["omega"] = 2**(1/3)
    p_zeta1["b1"] = 0.84987704
    p_zeta1["b2"] = 3.04033012073
    p_zeta1["b3"] = 0.0775730131248
    p_zeta1["b4"] = 7.57703592489
    p_zeta1["b5"] = get_b5(p_zeta1["omega"],p_zeta1["b3"])
    p_zeta1["c1"] = 0.91126873
    p_zeta1["c2"] =-0.0307957123308 
    p_zeta1["d1"] = 1.48658718
    p_zeta1["d2"] = 4.92684905511
    p_zeta1["d3"] = 0.0849387225179 
    p_zeta1["d4"] = 8.3269821188
    p_zeta1["d5"] = 0.218864952126
    p_zeta1["e1"] = 0.27454097
    p_zeta1["e2"] = 0.400994856555
    p_zeta1["e3"] = 2.88773194962
    p_zeta1["e4"] = 6.33499237092
    p_zeta1["e5"] = 24.823008753
    p1 = Parameters(p_zeta1)

    # spin interpolation 
    p_spin = {}
    # SS: sign of parameters is different as in the supp. mat. 
    p_spin["h1"] =3.18747258
    p_spin["h2"] =7.74662802
    p_spin["lambda1"] =1.85909536
    p_spin["lambda2"] = 0 
    p2 = Parameters(p_spin)

    return p0, p1, p2  

def get_a(theta): 
    """
        Get a. 
    """
    tmp1 = 0.610887*np.tanh(1.0/theta)
    tmp2 = (0.75 + 3.04363*theta**2 - 0.09227*theta**3 + 1.7035*theta**4)
    tmp3 = (1 + 8.31051*theta**2 + 5.1105*theta**4)
    a = tmp1 * tmp2 / tmp3
    return a

def get_dadtheta(theta):
    """
        Get da / dtheta. 
    """
    tmp1 = -0.00884515668249876*(20.442*theta**3 + 16.62102*theta)
    tmp2 = (1.7035*theta**4 - 0.09227*theta**3 + 3.04363*theta**2 + 0.75)
    tmp3 = np.tanh(1.0/theta)/(0.614944209200157*theta**4 + theta**2 + 0.12032955859508)**2
    denom = (5.1105*theta**4 + 8.31051*theta**2 + 1)
    tmp4 = 0.610887*(6.814*theta**3 - 0.27681*theta**2 + 6.08726*theta)*np.tanh(1.0/theta)/denom 
    thres = 0.001408 # 0.0025
    with np.errstate(over='ignore'):
        tmp5 = np.where((theta < 0.0025),0,-0.610887*(1.7035*theta**4 - 0.09227*theta**3 + 3.04363*theta**2 + 0.75)/(denom*theta**2*np.cosh(1.0/theta)**2))
    dadtheta=tmp1*tmp2*tmp3 + tmp4 + tmp5 
    return dadtheta

def get_b(theta,b1,b2,b3,b4,b5):
    """
        Get b. 
    """
    b = np.tanh(1/np.sqrt(theta)) * (b1 + b2*theta**2 + b3*theta**4) / (1 + b4*theta**2 + b5*theta**4)
    return b 

def get_dbdtheta(theta,b1,b2,b3,b4,b5):
    """
        Get db / dtheta. 
    """
    tmp1 = (2*b2*theta + 4*b3*theta**3)*np.tanh(1/np.sqrt(theta))/(b4*theta**2 + b5*theta**4 + 1)
    tmp2 = (2*b4*theta + 4*b5*theta**3)*(b1 + b2*theta**2 + b3*theta**4)*np.tanh(1/np.sqrt(theta))/(b4*theta**2 + b5*theta**4 + 1)**2
    with np.errstate(over='ignore'):
        tmp3 = np.where(theta < 0.001,0,(b1 + b2*theta**2 + b3*theta**4)/(2*(b4*theta**2 + b5*theta**4 + 1)*theta**(3/2)*np.cosh(1/np.sqrt(theta))**2))
    dbdtheta=tmp1 - tmp2 - tmp3 
    return dbdtheta

def get_e(theta,e1,e2,e3,e4,e5): 
    """ 
        Get e. 
    """
    e = np.tanh(1/theta)* (e1 + e2*theta**2 + e3*theta**4) / (1 + e4*theta**2 + e5*theta**4)
    return e

def get_dedtheta(theta,e1,e2,e3,e4,e5):
    """
        Get de / dtheta. 
    """
    tmp1 = (2*e2*theta + 4*e3*theta**3)*np.tanh(1/theta)/(e4*theta**2 + e5*theta**4 + 1)
    tmp2 = (2*e4*theta + 4*e5*theta**3)*(e1 + e2*theta**2 + e3*theta**4)*np.tanh(1/theta)/(e4*theta**2 + e5*theta**4 + 1)**2
    with np.errstate(over='ignore'):
        tmp3 = np.where(theta < 0.0025,0,(e1 + e2*theta**2 + e3*theta**4)/((e4*theta**2 + e5*theta**4 + 1)*theta**2*np.cosh(1/theta)**2))
    dedtheta=tmp1 - tmp2 - tmp3 
    return dedtheta

def get_c(theta,c1,c2,e1,e2,e3,e4,e5):
    """ 
        Get c. 
    """
    thres = 1e-6
    e = get_e(theta,e1,e2,e3,e4,e5)
    c = np.where(theta > thres,(c1 + c2*np.exp(-1/theta)) * e, c1*e)
    return c 

def get_dcdtheta(theta,c1,c2,e1,e2,e3,e4,e5):
    """
        Get dc / dtheta. 
    """
    e = get_e(theta,e1,e2,e3,e4,e5)
    tmp1 = c2*e*np.exp(-1/theta)/theta**2 
    tmp2 = (c1 + c2*np.exp(-1/theta))
    tmp3 = (2*e2*theta + 4*e3*theta**3)*np.tanh(1/theta)/(e4*theta**2 + e5*theta**4 + 1)
    tmp4 = (2*e4*theta + 4*e5*theta**3)*(e1 + e2*theta**2 + e3*theta**4)*np.tanh(1/theta)/(e4*theta**2 + e5*theta**4 + 1)**2
    with np.errstate(over='ignore'):
        tmp5 = np.where(theta < 0.0025,0,(e1 + e2*theta**2 + e3*theta**4)/((e4*theta**2 + e5*theta**4 + 1)*theta**2*np.cosh(1/theta)**2))
    dcdtheta=tmp1 + tmp2*(tmp3 - tmp4 - tmp5)
    return dcdtheta

def get_d(theta,d1,d2,d3,d4,d5):
    """
        Get d. 
    """
    d = np.tanh(1/np.sqrt(theta)) * (d1 + d2*theta**2 + d3*theta**4) / (1 + d4*theta**2 + d5*theta**4)
    return d 

def get_dddtheta(theta,d1,d2,d3,d4,d5):
    """ 
        Get dd / dtheta. 
    """
    tmp1 = (2*d2*theta + 4*d3*theta**3)*np.tanh(1/np.sqrt(theta))/(d4*theta**2 + d5*theta**4 + 1) 
    tmp2 = (2*d4*theta + 4*d5*theta**3)*(d1 + d2*theta**2 + d3*theta**4)*np.tanh(1/np.sqrt(theta))/(d4*theta**2 + d5*theta**4 + 1)**2
    with np.errstate(over='ignore'):
        tmp3 = np.where(theta < 0.001,0,(d1 + d2*theta**2 + d3*theta**4)/(2*(d4*theta**2 + d5*theta**4 + 1)*theta**(3/2)*np.cosh(1/np.sqrt(theta))**2))
    dddtheta=tmp1 - tmp2 - tmp3 
    return dddtheta

def get_fxc_zeta_params(rs,omega,a,b,c,d,e): 
    """
        Get fxc_zeta with explict parameters. 
    """
    fxc_zeta = -1/rs * (omega*a + np.sqrt(rs)*b + rs*c ) / (1 + np.sqrt(rs)*d + rs*e)
    return fxc_zeta 

def get_dfxc_zeta_paramsdtheta(rs,omega,a,b,c,d,e,dadtheta,dbdtheta,dcdtheta,dddtheta,dedtheta): 
    """ 
        Get dfxc_zeta / dtheta using explict parameters. 
    """
    dfxc_zetadtheta=(-np.sqrt(rs)*dddtheta - rs*dedtheta)*(-omega*a - b*np.sqrt(rs) - c*rs)/((d*np.sqrt(rs) + e*rs + 1)**2*rs) + (-omega*dadtheta - np.sqrt(rs)*dbdtheta - rs*dcdtheta)/((d*np.sqrt(rs) + e*rs + 1)*rs)
    return dfxc_zetadtheta

def get_fxc_zeta(rs,theta,p):
    """
        Get fxc_zeta using a parameters object. 
    """
    a = get_a(theta)
    b = get_b(theta,p.b1,p.b2,p.b3,p.b4,p.b5)
    e = get_e(theta,p.e1,p.e2,p.e3,p.e4,p.e5)
    c = get_c(theta,p.c1,p.c2,p.e1,p.e2,p.e3,p.e4,p.e5)
    d = get_d(theta,p.d1,p.d2,p.d3,p.d4,p.d5)
    fxc = get_fxc_zeta_params(rs,p.omega,a,b,c,d,e)
    return fxc 

def get_dfxc_zetadtheta(rs,theta,p): 
    """
        Get dfxc / dtheta. 
    """
    a = get_a(theta)
    b = get_b(theta,p.b1,p.b2,p.b3,p.b4,p.b5)
    e = get_e(theta,p.e1,p.e2,p.e3,p.e4,p.e5)
    c = get_c(theta,p.c1,p.c2,p.e1,p.e2,p.e3,p.e4,p.e5)
    d = get_d(theta,p.d1,p.d2,p.d3,p.d4,p.d5)
    dadtheta = get_dadtheta(theta)
    dbdtheta = get_dbdtheta(theta,p.b1,p.b2,p.b3,p.b4,p.b5)
    dcdtheta = get_dcdtheta(theta,p.c1,p.c2,p.e1,p.e2,p.e3,p.e4,p.e5)
    dddtheta = get_dddtheta(theta,p.d1,p.d2,p.d3,p.d4,p.d5)
    dedtheta = get_dedtheta(theta,p.e1,p.e2,p.e3,p.e4,p.e5)
    dfxcdtheta = get_dfxc_zeta_paramsdtheta(rs,p.omega,a,b,c,d,e,dadtheta,dbdtheta,dcdtheta,dddtheta,dedtheta)
    return dfxcdtheta

def get_theta0(theta,zeta): 
    """
        Get theta0. 
    """
    return theta*(1+zeta)**(2/3) 

def get_theta1(theta,zeta):
    """
        Get theta1. 
    """
    theta0 = get_theta0(theta,zeta) 
    theta1 = theta0*2**(-2/3)
    return theta1 

def get_lambda(rs,theta,p): 
    """
        get lambda. 
    """
    return p.lambda1 + p.lambda2*theta*rs**(1/2)

def get_h(rs,p): 
    """
        Get h. 
    """
    return (2/3 + p.h1*rs) / (1 + p.h2*rs) 

def get_alpha(rs,theta,p):
    """ 
        Get alpha. 
    """
    h = get_h(rs,p)
    lam = get_lambda(rs,theta,p) 
    return 2 - h*np.exp(-theta*lam)

def get_phi(rs,theta,zeta,p):
    """
        Get phi from rs, theta, and zeta. 
    """
    alpha = get_alpha(rs,theta,p)
    return ((1+zeta)**alpha + (1-zeta)**alpha -2) / (2**alpha-2)

def get_phi_T(rs,T,zeta,p):
    """
        Get phi from rs, T, and zeta. 
    """
    n = get_n(rs)
    theta = get_theta(T,n,zeta)
    alpha = get_alpha(rs,theta,p)
    return ((1+zeta)**alpha + (1-zeta)**alpha -2) / (2**alpha-2)

def get_fxc0(rs,theta,zeta):
    """
        Get fxc0.
    """
    p0, p1, p2 = get_gdsmfb_parameters() 
    theta0 = get_theta0(theta,zeta)
    fxc0 =get_fxc_zeta(rs,theta0,p0)
    return fxc0 

def get_fxc1(rs,theta,zeta):
    """
        Get fxc1. 
    """
    p0, p1, p2 = get_gdsmfb_parameters()
    theta1 = get_theta1(theta,zeta)
    fxc1 =get_fxc_zeta(rs,theta1,p1)
    return fxc1

def get_fxc(rs,theta,zeta,p0,p1,p2):
    """
        Get fxc utilizing rs,zeta, and theta. 
    """
    theta0 = get_theta0(theta,zeta)
    theta1 = get_theta1(theta,zeta)
    fxc0 = get_fxc_zeta(rs,theta0,p0)
    fxc1 = get_fxc_zeta(rs,theta1,p1)
    phi = get_phi(rs,theta0,zeta,p2)
    fxc = fxc0 + (fxc1 - fxc0) * phi 
    return fxc 

def get_fxc_nupndn(nup,ndn,T,p0,p1,p2):
    """
        Get fxc utilizing nup, ndn, and T. 
    """
    n = nup + ndn 
    zeta = (nup-ndn)/n 
    rs = get_rs_from_n(n)
    theta = get_theta(T,n,zeta)
    return get_fxc(rs,theta,zeta,p0,p1,p2)

def get_zeta(nup,ndn): 
    """
        Get zeta from nup and ndn.
    """
    return (nup-ndn)/(nup+ndn)

def get_dzetadnup(nup,ndn):
    """
        Get dzeta / dnup. 
    """
    dzetadnup=-(-ndn + nup)/(ndn + nup)**2 + 1/(ndn + nup)
    return dzetadnup

def get_dzetadndn(nup,ndn):
    """
        Get dzeta / dndn. 
    """
    dzetadndn=-(-ndn + nup)/(ndn + nup)**2 - 1/(ndn + nup)
    return dzetadndn

def get_drsdn(n):
    """
        Get drs / dn. 
    """
    drsdn = -6**(1/3)*(1/n)**(1/3)/(6*np.pi**(1/3)*n)
    return drsdn

def get_dtheta0dtheta(zeta):
    """
        Get dtheta0 / dtheta.
    """
    dtheta0dtheta=(zeta + 1)**(2/3)
    return dtheta0dtheta

def get_dtheta0dzeta(theta,zeta):
    """
        Get dtheta0 / dzeta.
    """
    dtheta0dzeta=2*theta/(3*(zeta + 1)**(1/3))
    return dtheta0dzeta

def get_theta_nup(T,nup):
    """
        Get theta from T and nup. 
    """
    k_fermi_sq = (6.0*np.pi*np.pi*nup)**(2.0/3.0)
    T_fermi = 1 / 2*k_fermi_sq
    theta = T/T_fermi
    return theta

def get_dthetadnup(T,nup):
    """
        Get dtheta / dnup. 
    """
    dthetadnup=-0.40380457618492*T/(np.pi**(4/3)*nup**(5/3))
    return dthetadnup

def get_dfxc_zeta_paramsdrs(rs,omega,a,b,c,d,e):
    """
        Get dfxc / drs with explicit parameters. 
    """
    dfxc_zetadrs=(-b/(2*np.sqrt(rs)) - c)/(rs*(d*np.sqrt(rs) + e*rs + 1)) + (-d/(2*np.sqrt(rs)) - e)*(-a*omega - b*np.sqrt(rs) - c*rs)/(rs*(d*np.sqrt(rs) + e*rs + 1)**2) - (-a*omega - b*np.sqrt(rs) - c*rs)/(rs**2*(d*np.sqrt(rs) + e*rs + 1))
    return dfxc_zetadrs

def get_dfxc_zetadrs(rs,theta,p):
    """
        Get dfxc / drs utilizing a parameters object.
    """
    a = get_a(theta)
    b = get_b(theta,p.b1,p.b2,p.b3,p.b4,p.b5)
    e = get_e(theta,p.e1,p.e2,p.e3,p.e4,p.e5)
    c = get_c(theta,p.c1,p.c2,p.e1,p.e2,p.e3,p.e4,p.e5)
    d = get_d(theta,p.d1,p.d2,p.d3,p.d4,p.d5)
    dfxc_zetadrs=get_dfxc_zeta_paramsdrs(rs,p.omega,a,b,c,d,e)
    return dfxc_zetadrs

def get_dfxcdnup_params(nup,ndn,T,p0,p1,p2):
    """
        Get dfxc / dnup.
    """
    n = nup + ndn
    zeta = (nup-ndn)/n
    rs = get_rs_from_n(n)
    theta = get_theta(T,n,zeta)
    theta0 = get_theta0(theta,zeta)
    fxc0 =get_fxc_zeta(rs,theta0,p0)
    theta1 = get_theta1(theta,zeta)
    fxc1 = get_fxc_zeta(rs,theta1,p1)
    phi = get_phi(rs,theta0,zeta,p2)
    
    dndnup = 1 
    dzetadnup = get_dzetadnup(nup,ndn)
    drsdn = get_drsdn(n)
    dfxc0drs = get_dfxc_zetadrs(rs,theta,p0)
    dfxc1drs = get_dfxc_zetadrs(rs,theta,p1)

    dfxc0dtheta0 = get_dfxc_zetadtheta(rs,theta0,p0)
    dfxc1dtheta1 = get_dfxc_zetadtheta(rs,theta1,p1)

    dtheta0dtheta = get_dtheta0dtheta(zeta)
    dthetadnup = get_dthetadnup(T,nup)
    dtheta0dzeta = get_dtheta0dzeta(theta0,zeta)
    dtheta1dtheta0 = 2**(-2/3)

    dphidrs = get_dphidrs(rs,theta0,zeta,p2)
    dphidtheta = get_dphidtheta(rs,theta0,zeta,p2)
    dphidzeta = get_dphidzeta(rs,theta0,zeta,p2) 

    dfxc0a = dfxc0drs*dndnup*drsdn
    dfxc1a = dfxc1drs*dndnup*drsdn
    dfxc0b = dfxc0dtheta0*(dtheta0dtheta*dthetadnup + dtheta0dzeta*dzetadnup)
    dfxc1b = dfxc1dtheta1*dtheta1dtheta0*(dtheta0dtheta*dthetadnup + dtheta0dzeta*dzetadnup)
    dphi = (dndnup*dphidrs*drsdn + dphidtheta*dthetadnup + dphidzeta*dzetadnup)
    dfxcdnup=dfxc0a + dfxc0b - phi*(dfxc0a + dfxc0b - dfxc1a - dfxc1b) - (fxc0 - fxc1)*dphi
    fxc = get_fxc_nupndn(nup,ndn,T,p0,p1,p2)
    return dfxcdnup # fxc + dfxcdnup 

def get_dfxcdndn_params(nup,ndn,T,p0,p1,p2):
    """
        Get dfxc / dndn. 
    """
    n = nup + ndn
    zeta = (nup-ndn)/n
    rs = get_rs_from_n(n)
    theta = get_theta(T,n,zeta)
    theta0 = get_theta0(theta,zeta)
    fxc0 =get_fxc_zeta(rs,theta0,p0)
    theta1 = get_theta1(theta,zeta)
    fxc1 = get_fxc_zeta(rs,theta1,p1)
    phi = get_phi(rs,theta0,zeta,p2)
 
    dndndn = 1
    dzetadndn = get_dzetadndn(nup,ndn)
    drsdn = get_drsdn(n)
    dfxc0drs = get_dfxc_zetadrs(rs,theta,p0)
    dfxc1drs = get_dfxc_zetadrs(rs,theta,p1)

    dfxc0dtheta0 = get_dfxc_zetadtheta(rs,theta0,p0)
    dfxc1dtheta1 = get_dfxc_zetadtheta(rs,theta1,p1)

    dtheta0dtheta = get_dtheta0dtheta(zeta)
    dthetadndn=0
    dtheta0dzeta = get_dtheta0dzeta(theta0,zeta)
    dtheta1dtheta0 = 2**(-2/3)

    dphidrs = get_dphidrs(rs,theta0,zeta,p2)
    dphidtheta = get_dphidtheta(rs,theta0,zeta,p2)
    dphidzeta = get_dphidzeta(rs,theta0,zeta,p2)

    dfxc0a = dfxc0drs*dndndn*drsdn
    dfxc1a = dfxc1drs*dndndn*drsdn
    dfxc0b = dfxc0dtheta0*(dtheta0dtheta*dthetadndn + dtheta0dzeta*dzetadndn)
    dfxc1b = dfxc1dtheta1*dtheta1dtheta0*(dtheta0dtheta*dthetadndn + dtheta0dzeta*dzetadndn)
    dphi = (dndndn*dphidrs*drsdn + dphidtheta*dthetadndn + dphidzeta*dzetadndn)
    dfxcdndn=dfxc0a + dfxc0b - phi*(dfxc0a + dfxc0b - dfxc1a - dfxc1b) - (fxc0 - fxc1)*dphi
    fxc = get_fxc_nupndn(nup,ndn,T,p0,p1,p2)
    return dfxcdndn #fxc + dfxcdndn

def get_rs_from_n(n): 
    """ Get rs from n."""
    rs = (3 / (4 * np.pi * n)) ** (1 / 3)
    return rs 

def get_n(rs):
    """ Get n from rs."""
    return 1.0 / ( 4.0 * np.pi / 3.0 *rs**3.0)

def get_dhdrs(rs,p):
    """ Get dh / drs."""
    dhdrs=p.h1/(p.h2*rs + 1) - p.h2*(p.h1*rs + 2/3)/(p.h2*rs + 1)**2
    return dhdrs 

def get_dlamdrs(rs,theta,p):
    """ Get dlam / drs."""
    dlamdrs=p.lambda2*theta/(2*np.sqrt(rs))
    return dlamdrs 

def get_dalphadrs(rs,theta,p):
    """ Get dalpha / drs."""
    h = get_h(rs,p)
    lam = get_lambda(rs,theta,p)
    dhdrs = get_dhdrs(rs,p) 
    dlamdrs = get_dlamdrs(rs,theta,p)
    dalphadrs=-dhdrs*np.exp(-theta*lam) + dlamdrs*theta*h*np.exp(-theta*lam)
    return dalphadrs

def get_dlamdtheta(rs,theta,p):
    """ Get dlam / dtheta."""
    dlamdtheta=p.lambda2*np.sqrt(rs)
    return dlamdtheta

def get_dalphadtheta(rs,theta,p):
    """ Get dalpha / dtheta."""
    h = get_h(rs,p)
    lam = get_lambda(rs,theta,p)
    dlamdtheta = get_dlamdtheta(rs,theta,p) 
    dalphadtheta=-(-dlamdtheta*theta - lam)*h*np.exp(-theta*lam)
    return dalphadtheta

def get_zeta_rs(rs,nup,ndn):
    """ Get zeta from rs, nup, and ndn."""
    return (nup-ndn)/(get_n(rs))

def get_dzetadrs(rs,nup,ndn):
    """ Get dzeta / drs."""
    dzetadrs=4.0*np.pi*(-ndn + nup)*rs**2.0
    return dzetadrs

def get_dphidrs(rs,theta,zeta,p):
    """ Get dphi / drs."""
    thres = 1e-15
    n = get_n(rs)
    ndn = n*(1+zeta)/2
    nup = n*(1-zeta)/2
    alpha = get_alpha(rs,theta,p)
    duv = ((1 - zeta)**alpha*np.log(1 - zeta +thres) + (zeta + 1)**alpha*np.log(zeta + 1))*(2**alpha - 2)
    udv = ((1 - zeta)**alpha + (zeta + 1)**alpha - 2)*(2**alpha)*np.log(2)
    vv = (2**alpha - 2)**2
    dalphadrs = get_dalphadrs(rs,theta,p)
    dphidrs = (duv - udv)*dalphadrs/vv
    return dphidrs 

def get_dphidtheta(rs,theta,zeta,p):
    """ Get dphi / dtheta."""
    thres = 1e-15
    n = get_n(rs)
    ndn = n*(1+zeta)/2
    nup = n*(1-zeta)/2
    alpha = get_alpha(rs,theta,p)
    dalphadtheta = get_dalphadtheta(rs,theta,p)
    duv = ((1 - zeta)**alpha*np.log(1 - zeta +thres) + (zeta + 1)**alpha*np.log(zeta + 1))*(2**alpha - 2)
    udv = ((1 - zeta)**alpha + (zeta + 1)**alpha - 2)*(2**alpha)*np.log(2)
    vv = (2**alpha - 2)**2
    dalphadtheta = get_dalphadtheta(rs,theta,p)
    dphidrs = (duv - udv)*dalphadtheta/vv
    return dphidrs

def get_dphidzeta(rs,theta,zeta,p):
    """ Get dphi / dzeta."""
    alpha = get_alpha(rs,theta,p)
    # Handle divisions by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        dphidzeta=(alpha*(zeta + 1)**alpha/(zeta + 1) - alpha*(1 - zeta)**alpha/(1 - zeta))/(2**alpha - 2)
    dphidzeta = np.nan_to_num(dphidzeta, nan=0, posinf=0, neginf=0)
    return dphidzeta

def get_theta(T,n,zeta):
    """ Reduced temperature 
        
    Calculates the reduced temperature 

    theta = T / T_Fermi

    Reference: Phys. Rev. Lett. 119, 135001.

    Args: 
        T: Absolute temperature in Hartree. 
        n: Real-space electronic temperature. 
        zeta: Relative spin polarization.

    Keyword Args:
        **kwargs: Throwaway arguments.

    Returns: 
        Reduced temperature. 
    """
    n_up = 0.5 * n * (1.0 + zeta)
    k_fermi_sq = (6.0*np.pi*np.pi*n_up)**(2.0/3.0)
    T_fermi = 1 / 2*k_fermi_sq
    theta = T/T_fermi
    return theta

def get_T(theta,n,zeta):
    """ Absolute temperature 

    Calculates the absolute temperature

    T = theta * T_Fermi 

    Reference: Phys. Rev. Lett. 119, 135001.

    Args:
        theta: reduced temperature.
        n: Real-space electronic temperature.
        zeta: Relative spin polarization.

    Keyword Args:
        **kwargs: Throwaway arguments.

    Returns:
        Reduced temperature.
    """
    rs = get_rs_from_n(n) 
    n_up = 0.50 * n * (1.0 + zeta)
    k_fermi_sq = (6.0*np.pi*np.pi*n_up)**(2.0/3.0)
    T_fermi = 1 / 2*k_fermi_sq
    T = theta * T_fermi
    return T


def lda_xc_gdsmfb_spin(n, zeta, **kwargs):
    """ GDSMFB exchange-correlation functional (spin-polarized).

    Exchange and correlation connot be separated.

    Reference: Phys. Rev. Lett. 119, 135001.

    Args:
        n: Real-space electronic density.
        zeta: Relative spin polarization.

    Keyword Args:
        T: Temperature. 

    Returns:
        GDSMFB exchange-correlation energy density and potential.
    """
    T = kwargs.get("T")
    ndn = n*(1-zeta)/2
    nup = n*(1+zeta)/2
    rs = get_rs_from_n(n)
    theta = get_theta(T,n,zeta)
    theta0 = get_theta0(theta,zeta)
    theta1 = get_theta1(theta,zeta)

    # parameters
    p0, p1, p2 = get_gdsmfb_parameters()

    # fxc
    fxc = get_fxc(rs,theta,zeta,p0,p1,p2)

    # dfxcdnup
    dfxcdnup = get_dfxcdnup_params(nup,ndn,T,p0,p1,p2)

    # dfxcdndn
    dfxcdndn = get_dfxcdndn_params(nup,ndn,T,p0,p1,p2)

    return fxc, np.array([dfxcdnup,dfxcdndn])*n+fxc, None 

