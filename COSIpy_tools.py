import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
plt.style.use('thomas')
import astropy.units as u
import sys
import time
import scipy.interpolate as interpol
import matplotlib
from matplotlib import ticker, cm
from scipy.ndimage import gaussian_filter
from tqdm import tqdm_notebook as tqdm
import os, glob 

import pandas as pd

# import ROOT/MEGAlib to work with python
import ROOT as M
# Load MEGAlib into ROOT
M.gSystem.Load("$(MEGAlib)/lib/libMEGAlib.so")
# Initialize MEGAlib
G = M.MGlobal()
G.Initialize()


def minmin(x):
    """
    Return array shifted to zero
    :param: x   Input array
    """
    return x - x.min()


def maxmax(x):
    """
    Return array shifted to maximum
    :param: x   Input array
    """
    return x.max() - x


def minmax(x):
    """
    Return minimum and maximum of array
    :param: x   Input array
    """
    return np.array([x.min(),x.max()])


def find_nearest(array, value):
    """
    Find nearest index for value in array (where for non-existent values)
    :param: array   Input array
    :param: value   value to search for
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def verb(q,text):
    """
    Print text of q is True
    :param:   q (boolean)
    :param:   text
    """
    if q:
        print(text)

        
def polar2cart(ra,dec):
    """
    Coordinate transformation of ra/dec (lon/lat) [phi/theta] polar/spherical coordinates
    into cartesian coordinates
    :param: ra   angle in deg
    :param: dec  angle in deg
    """
    x = np.cos(np.deg2rad(ra)) * np.cos(np.deg2rad(dec))
    y = np.sin(np.deg2rad(ra)) * np.cos(np.deg2rad(dec))
    z = np.sin(np.deg2rad(dec))
    
    return np.array([x,y,z])


def cart2polar(vector):
    """
    Coordinate transformation of cartesian x/y/z values into spherical (deg)
    :param: vector   vector of x/y/z values
    """
    ra = np.arctan2(vector[1],vector[0]) 
    dec = np.arcsin(vector[2])
    
    return np.rad2deg(ra), np.rad2deg(dec)


def construct_scy(scx_l, scx_b, scz_l, scz_b):
    """
    Construct y-coordinate of spacecraft/balloon given x and z directions
    Note that here, z is the optical axis
    :param: scx_l   longitude of x direction
    :param: scx_b   latitude of x direction
    :param: scz_l   longitude of z direction
    :param: scz_b   latitude of z direction
    """
    x = polar2cart(scx_l, scx_b)
    z = polar2cart(scz_l, scz_b)
    
    return cart2polar(np.cross(z,x,axis=0))
        
        
def read_COSI_DataSet(dor):
    """
    Reads in all MEGAlib .tra (or .tra.gz) files in given directory 'dor'
    If full flight/mission data set is read in, e.g. from /FlightData_Processed_v4/ with standard names,
    the data set is split up into days and hours of observations. Otherwise, everything is lumped together.
    Returns COSI data set as a dictionary of the form
    COSI_DataSet = {'Day':trafiles[i][StrBeg:StrBeg+StrLen],
                    'Energies':erg,
                    'TimeTags':tt,
                    'Xpointings':np.array([lonX,latX]).T,
                    'Ypointings':np.array([lonY,latY]).T,
                    'Zpointings':np.array([lonZ,latZ]).T,
                    'Phi':phi,
                    'Chi local':chi_loc,
                    'Psi local':psi_loc,
                    'Distance':dist,
                    'Chi galactic':chi_gal,
                    'Psi galactic':psi_gal}
    :param: dor    Path of directory in which data set is stored
    """
    
    # Search for files in directory
    trafiles = []
    for dirpath, subdirs, files in os.walk(dor):
        for file in files:
            if glob.fnmatch.fnmatch(file,"*.tra*"):
                trafiles.append(os.path.join(dirpath, file))

    # sort files (only with standard names)
    trafiles = sorted(trafiles)

    # string lengths to get day as key
    StrBeg = len(dor)+len('OP_')
    StrLen = 6

    # read COSI data
    COSI_Data = []
    
    # loop over all tra files
    for i in tqdm(range(len(trafiles))):
        
        Reader = M.MFileEventsTra()
        if Reader.Open(M.MString(trafiles[i])) == False:
            print("Unable to open file " + trafiles[i] + ". Aborting!")
            quit()
            
        erg = []   # Compton energy
        tt = []    # Time tag
        et = []    # Event Type
        latX = []  # latitude of X direction of spacecraft
        lonX = []  # lontitude of X direction of spacecraft
        latZ = []  # latitude of Z direction of spacecraft
        lonZ = []  # longitude of Z direction of spacecraft
        phi = []   # Compton scattering angle
        chi_loc = [] # measured data space angle chi
        psi_loc = [] # measured data space angle psi
        dist = []    # First lever arm
        chi_gal = [] # measured gal angle chi (lon direction)
        psi_gal = [] # measured gal angle psi (lat direction)

        while True:
            Event = Reader.GetNextEvent()
            if not Event:
                break
            if Event.GetEventType() == M.MPhysicalEvent.c_Compton:
                # all calculations and definitions taken from /MEGAlib/src/response/src/MResponseImagingBinnedMode.cxx
                erg.append(Event.Ei()) # Total Compton Energy
                tt.append(Event.GetTime().GetAsSeconds()) # Time tag in seconds since ...
                et.append(Event.GetEventType()) # Event type (0 = Compton, 4 = Photo)
                latX.append(Event.GetGalacticPointingXAxisLatitude()) # x axis of space craft pointing at GAL latitude
                lonX.append(Event.GetGalacticPointingXAxisLongitude()) # x axis of space craft pointing at GAL longitude
                latZ.append(Event.GetGalacticPointingZAxisLatitude()) # z axis of space craft pointing at GAL latitude
                lonZ.append(Event.GetGalacticPointingZAxisLongitude()) # z axis of space craft pointing at GAL longitude
                phi.append(Event.Phi()) # Compton scattering angle
                chi_loc.append((-Event.Dg()).Phi())
                psi_loc.append((-Event.Dg()).Theta())
                dist.append(Event.FirstLeverArm())
                chi_gal.append((Event.GetGalacticPointingRotationMatrix()*Event.Dg()).Phi())
                psi_gal.append((Event.GetGalacticPointingRotationMatrix()*Event.Dg()).Theta())

        erg = np.array(erg)
        tt = np.array(tt)
        et = np.array(et)
        latX = np.array(latX)
        lonX = np.array(lonX)
        lonX[lonX > np.pi] -= 2*np.pi
        latZ = np.array(latZ)
        lonZ = np.array(lonZ)
        lonZ[lonZ > np.pi] -= 2*np.pi
        phi = np.array(phi)
        chi_loc = np.array(chi_loc)
        chi_loc[chi_loc < 0] += 2*np.pi
        psi_loc = np.array(psi_loc)
        dist = np.array(dist)
        chi_gal = np.array(chi_gal)
        psi_gal = np.array(psi_gal)
        # construct Y direction from X and Z direction
        lonlatY = construct_scy(np.rad2deg(lonX),np.rad2deg(latX),np.rad2deg(lonZ),np.rad2deg(latZ))
        lonY = np.deg2rad(lonlatY[0])
        latY = np.deg2rad(lonlatY[1])
        # avoid negative zeros
        chi_loc[np.where(chi_loc == 0.0)] = np.abs(chi_loc[np.where(chi_loc == 0.0)])
        
        # make observation dictionary
        COSI_DataSet = {'Day':trafiles[i][StrBeg:StrBeg+StrLen],
                        'Energies':erg,
                        'TimeTags':tt,
                        'Xpointings':np.array([lonX,latX]).T,
                        'Ypointings':np.array([lonY,latY]).T,
                        'Zpointings':np.array([lonZ,latZ]).T,
                        'Phi':phi,
                        'Chi local':chi_loc,
                        'Psi local':psi_loc,
                        'Distance':dist,
                        'Chi galactic':chi_gal,
                        'Psi galactic':psi_gal}
    
        COSI_Data.append(COSI_DataSet)

    return COSI_Data


def hourly_tags(COSI_Data):
    """
    Get COSI data reformatted to one data set per hour in each day of the observation
    (Basically only useful if not single observations are used)
    Output is a dictionary of the form:
    data = {'Day':COSI_Data[i]['Day'],
            'Hour':h,
            'Indices':tdx}
    :param: COSI_Data   dictionary from read_COSI_DataSet
    """
    # time conversions
    s2d = 1./86400
    s2h = 1./3600
    # number of hours per day
    n_hours = 24
    
    # fill data
    data = []

    for i in range(len(COSI_Data)):
        for h in range(n_hours):
            tdx = np.where( (minmin(COSI_Data[i]['TimeTags'])*s2h >= h) &
                            (minmin(COSI_Data[i]['TimeTags'])*s2h <= h+1) )

            tmp_data = {'Day':COSI_Data[i]['Day'],
                        'Hour':h,
                        'Indices':tdx}
            
            data.append(tmp_data)
    
    return data
        

def hourly_binning(COSI_Data):
    """
    Get total times, counts, and average observations per hour of observation
    (Note that the mean of different angles over time doesnt make sense, and should only be used for illustration purpose!)
    Output is a dictionary of the form:
    data = {'Times':times,
            'Counts':counts,
            'GLons':glons,
            'GLats':glats}
    :param: COSI_Data   dictionary from read_COSI_DataSet
    """
    
    # initialise sequential binning
    tmp = np.histogram(COSI_Data[0]['TimeTags'],bins=24)
    times = tmp[1][0:-1]
    counts = tmp[0]
    # add nonsense start array (remove later)
    gcoords = np.array([[-99],[-99]])
    # loop over hours
    for t in range(len(tmp[0])):
        gcoords = np.concatenate([gcoords,
                                  np.nanmean(COSI_Data[0]['Zpointings'][np.where((COSI_Data[0]['TimeTags'] >= tmp[1][t]) & 
                                                                                 (COSI_Data[0]['TimeTags'] <= tmp[1][t+1]))[0],:],
                                                                                 axis=0).reshape(2,1)],axis=1)
    glons = gcoords[0,:]
    glats = gcoords[1,:]
    glons = np.delete(glons,0) # get rid of -99
    glats = np.delete(glats,0) # get rid of -99
    
    # loop over remaining days
    for i in range(len(COSI_Data)-1):
        # check for unexpected time tags
        bad_idx = np.where(COSI_Data[i+1]['TimeTags'] < COSI_Data[i]['TimeTags'][-1])
        if len(bad_idx[0]) > 0:
            for key in COSI_Data[i+1].keys():
                if key != 'Day':
                    COSI_Data[i+1][key] = np.delete(COSI_Data[i+1][key],bad_idx,axis=0)
        # continue as before
        tmp = np.histogram(COSI_Data[i+1]['TimeTags'],bins=24)
        times = np.concatenate((times,tmp[1][0:-1]))
        counts = np.concatenate((counts,tmp[0]))
    
        gcoords = np.array([[-99],[-99]])
        for t in range(len(tmp[0])):
            gcoords = np.concatenate([gcoords,
                                      np.nanmean(COSI_Data[i+1]['Zpointings'][np.where((COSI_Data[i+1]['TimeTags'] >= tmp[1][t]) &
                                                                                       (COSI_Data[i+1]['TimeTags'] <= tmp[1][t+1]))[0],:],
                                                                                       axis=0).reshape(2,1)],axis=1)
        glons_tmp = gcoords[0,:]
        glats_tmp = gcoords[1,:]
        glons_tmp = np.delete(glons_tmp,0)
        glats_tmp = np.delete(glats_tmp,0)
        glons = np.concatenate([glons,glons_tmp])
        glats = np.concatenate([glats,glats_tmp])

    data = {'Times':times,
            'Counts':counts,
            'GLons':glons,
            'GLats':glats}

    return data

        
def FISBEL(n_bins,lon_shift,verbose=False):
    """
    MEGAlib's FISBEL spherical axis binning
    /MEGAlib/src/global/misc/src/MBinnerFISBEL.cxx
    Used to make images with more information (pixels) in the centre, and less at higher latitudes
    // CASEBALL - Constant area, squares at equator, borders along latitude & longitude

    // Rules:
    // (1) constant area
    // (2) squarish at equator
    // (3) All borders along latitude & longitude lines
    // (4) latitude distance close to equal
    
    Don't know where "lon_shift" is actually required, keeping it for the moment
    :param: n_bins      number of bins to populate the sky (typically 1650 (~5 deg) or 4583 (~3 deg))
    :param: lon_shift   ???
    :option: verbose    True = more verbose output
    
    Returns CoordinatePairs and respective Binsizes
    """
    
    # if only one bin, full sky in one bin
    if (n_bins == 1):
        LatitudeBinEdges = [0,np.pi]
        LongitudeBins = [1]
    else:
        FixBinArea = 4*np.pi/n_bins
        SquareLength = np.sqrt(FixBinArea)
        
        n_collars = np.int((np.pi/SquareLength-1)+0.5) + 2
        # -1 for half one top AND Bottom, 0.5 to round to next integer
        # +2 for the half on top and bottom
        
        verb(verbose,'Number of bins: %4i' % n_bins)
        verb(verbose,'Fix bin area: %6.3f' % FixBinArea)
        verb(verbose,'Square length: %6.3f' % SquareLength)
        verb(verbose,'Number of collars: %4i' % n_collars)
        
        LongitudeBins = np.zeros(n_collars)
        LatitudeBinEdges = np.zeros(n_collars+1)
        
        # Top and bottom first
        LatitudeBinEdges[0] = 0
        LatitudeBinEdges[n_collars] = np.pi

        # Start on top and bottom with a circular region:
        LongitudeBins[0] = 1
        LongitudeBins[n_collars - 1] = LongitudeBins[0]

        LatitudeBinEdges[1] = np.arccos(1 - 2.0/n_bins)
        LatitudeBinEdges[n_collars - 1] = np.pi - LatitudeBinEdges[1]
        
        # now iterate over remaining bins
        for collar in range(1,np.int(np.ceil(n_collars/2))):
            UnusedLatitude = LatitudeBinEdges[n_collars-collar] - LatitudeBinEdges[collar]
            UnusedCollars = n_collars - 2*collar
            
            NextEdgeEstimate = LatitudeBinEdges[collar] + UnusedLatitude/UnusedCollars
            NextBinsEstimate = 2*np.pi * (np.cos(LatitudeBinEdges[collar]) - np.cos(NextEdgeEstimate)) / FixBinArea
            
            # roundgind
            NextBins = np.int(NextBinsEstimate+0.5)
            NextEdge = np.arccos(np.cos(LatitudeBinEdges[collar]) - NextBins*FixBinArea/2/np.pi)
            
            # insert at correct position
            LongitudeBins[collar] = NextBins
            if (collar != n_collars/2):
                LatitudeBinEdges[collar+1] = NextEdge
            LongitudeBins[n_collars-collar-1] = NextBins
            if (collar != n_collars/2):
                LatitudeBinEdges[n_collars-collar-1] = np.pi - NextEdge
          
    LongitudeBinEdges = []
    for nl in LongitudeBins:
        if (nl == 1):
            LongitudeBinEdges.append(np.array([0,2*np.pi]))
        else:
            n_lon_edges = nl+1
            LongitudeBinEdges.append(np.linspace(0,2*np.pi,n_lon_edges))
    
    #verb(verbose,'LatitudeBins: %4i' % n_collars)
    #verb(verbose,'LatitudeBinEdges: %6.3f' % LatitudeBinEdges)
    #verb(verbose,'LongitudeBins: %4i' % LongitudeBins)
    #verb(verbose,'LongitudeBinEdges: %6.3f' % np.array(LongitudeBinEdges))
    
    CoordinatePairs = []
    Binsizes = []
    for c in range(n_collars):
        for l in range(np.int(LongitudeBins[c])):
            CoordinatePairs.append([np.mean(LatitudeBinEdges[c:c+2]),np.mean(LongitudeBinEdges[c][l:l+2])])
            Binsizes.append([np.diff(LatitudeBinEdges[c:c+2]),np.diff(LongitudeBinEdges[c][l:l+2])])
     
    return CoordinatePairs,Binsizes


def get_binned_data(COSI_Data,tdx,pp,bb,ll,dpp,dbb,dll):
    """
    Re-format COSI data set into phi, psi, and chi (last two FISBELed) bins.
    Here assuming full data set with 47 days and 24 hours, requires FISBEL and phi bins and edges
    to be set up with COSIpy_tools.FISBEL()
    :param: COSI_Data   COSI data set dictionary from read_COSI_DataSet()
    :param: tdx         Time tag indices as hourly binned dictionary from hourly_tags()
    :param: pp          PHI bins from FISBEL
    :param: bb          PSI bins from FISBEL
    :param: ll          CHI bins from FISBEL
    :param: dpp         PHI bin sizes
    :param: dbb         PSI bin sizes
    :param: dll         CHI bin sizes
    
    Returns 4D array of shape (47,24,36,1650) with 47 days, 24 hours, 36 PHI, abd 1650 FISBEL bins
    """
    # init data array
    binned_data = np.zeros((47,24,36,1650))
    
    # not really used now; played with tolerance for bin edges (now np.around >> very slow)
    tol = 1e-6
    
    # loop over days, hours, phi, psi-chi-fisbel
    for d in tqdm(range(47)):
        for h in range(24):
            for p in range(len(pp)):
                for c in range(len(ll)):
                    binned_data[d,h,p,c] = len(np.where((COSI_Data[d]['Phi'][tdx[d,h]['Indices']] >= np.around(pp[p]-dpp[p]/2,6)) &
                                                        (COSI_Data[d]['Phi'][tdx[d,h]['Indices']] < np.around(pp[p]+dpp[p]/2,6)) &
                                                        (COSI_Data[d]['Psi local'][tdx[d,h]['Indices']] >= np.around(bb[c]-dbb[c]/2,6)) &
                                                        (COSI_Data[d]['Psi local'][tdx[d,h]['Indices']] < np.around(bb[c]+dbb[c]/2,6)) &
                                                        (COSI_Data[d]['Chi local'][tdx[d,h]['Indices']] >= np.around(ll[c]-dll[c]/2,6)) &
                                                        (COSI_Data[d]['Chi local'][tdx[d,h]['Indices']] < np.around(ll[c]+dll[c]/2,6)))[0])
                    
    #might be faster with some tools, don't know yet
    return binned_data


def get_binned_data_complete(COSI_Data,pp,bb,ll,dpp,dbb,dll):
    """
    Same as get_binned_data() but for data set without separate time information (everything lumped together)
    :param: COSI_Data   COSI data set dictionary from read_COSI_DataSet()
    :param: pp          PHI bins from FISBEL
    :param: bb          PSI bins from FISBEL
    :param: ll          CHI bins from FISBEL
    :param: dpp         PHI bin sizes
    :param: dbb         PSI bin sizes
    :param: dll         CHI bin sizes
    
    Returns 2D array of shape (36,1650) with 36 PHI, abd 1650 FISBEL bins
    """
    # init data array
    binned_data = np.zeros((36,1650))
    
    # loop over phi and psi-chi-fisbel
    for p in range(len(pp)):
        for c in range(len(ll)):
            # darn np.around
            idx = np.where((COSI_Data[0]['Phi'] >= np.around(pp[p]-dpp[p]/2,6)) &
                                            (COSI_Data[0]['Phi'] < np.around(pp[p]+dpp[p]/2,6)) &
                                            (COSI_Data[0]['Psi local'] >= np.around(bb[c]-dbb[c]/2,6)) &
                                            (COSI_Data[0]['Psi local'] < np.around(bb[c]+dbb[c]/2,6)) &
                                            (COSI_Data[0]['Chi local'] >= np.around(ll[c]-dll[c]/2,6)) &
                                            (COSI_Data[0]['Chi local'] < np.around(ll[c]+dll[c]/2,6)))[0]
            
            binned_data[p,c] = len(idx)

    return binned_data


def get_binned_data_oneday(COSI_Data,tdx,pp,bb,ll,dpp,dbb,dll):
    """
    Same as get_binned_data() but for data set with only one day (24 hours)
    :param: COSI_Data   COSI data set dictionary from read_COSI_DataSet()
    :param: tdx         Time tag indices as hourly binned dictionary from hourly_tags() as (1,24) array
    :param: pp          PHI bins from FISBEL
    :param: bb          PSI bins from FISBEL
    :param: ll          CHI bins from FISBEL
    :param: dpp         PHI bin sizes
    :param: dbb         PSI bin sizes
    :param: dll         CHI bin sizes
    
    Returns 3D array of shape (24,36,1650) with 36 PHI, abd 1650 FISBEL bins
    """
    # init data array
    binned_data = np.zeros((24,36,1650))
    # same as above
    tol = 1e-6
    # loops as above
    for h in tqdm(range(24)):
        for p in range(len(pp)):
            for c in range(len(ll)):
                binned_data[h,p,c] = len(np.where((COSI_Data[0]['Phi'][tdx[0,h]['Indices']] >= np.around(pp[p]-dpp[p]/2,6)) &
                                                  (COSI_Data[0]['Phi'][tdx[0,h]['Indices']] < np.around(pp[p]+dpp[p]/2,6)) &
                                                  (COSI_Data[0]['Psi local'][tdx[0,h]['Indices']] >= np.around(bb[c]-dbb[c]/2,6)) &
                                                  (COSI_Data[0]['Psi local'][tdx[0,h]['Indices']] < np.around(bb[c]+dbb[c]/2,6)) &
                                                  (COSI_Data[0]['Chi local'][tdx[0,h]['Indices']] >= np.around(ll[c]-dll[c]/2,6)) &
                                                  (COSI_Data[0]['Chi local'][tdx[0,h]['Indices']] < np.around(ll[c]+dll[c]/2,6)))[0])
                
    return binned_data


def plot_FISBEL(ll,bb_in,dll,dbb,values,colorbar=False,projection=False):
    """
    Plot FISBEL-binned data given the definition of FISBEL bins and the corresponding 1D array of values
    :param: ll         azimuth,longitude,psi bin centers
    :param: bb_in      zenith,latitude,theta bin centers
    :param: dll        azimuth,longitude,psi bin sizes
    :param: dbb        zenith,latitude,theta bin sizes
    :param: values     array of values to plot as color (3rd dimension)
    :option: colorbar  True = add colorbar to plot
    
    """
    # change initial FISBEL definition of bb to avoid random graphic issues (mainly in projection)
    bb = bb_in - np.pi
    bb = np.abs(bb)

    # set up figure
    plt.figure(figsize=(16,9))
    if projection:
        plt.subplot(projection=projection)
        bb -= np.pi/2
    
    # make colorbar in viridis according to entry values
    tmp_data = values
    cmap = plt.cm.viridis
    norm = matplotlib.colors.Normalize(vmin=tmp_data.min(), vmax=tmp_data.max())

    # plot points only
    fig = plt.scatter(ll,bb,s=0.5,zorder=3000,c=tmp_data)
    if colorbar==True:
        plt.colorbar()

    plt.scatter(ll,bb,s=0.5,zorder=3001)
    
    # loop over FISBEL bins to add coloured tiles 
    for i in range(len(bb)):
        # no idea how to makes this more efficient, but takes for ever
        plt.fill_between((ll[i]+np.array([-dll[i],+dll[i],+dll[i],-dll[i],-dll[i]])/2.),
                         (bb[i]+np.array([-dbb[i],-dbb[i],+dbb[i],+dbb[i],-dbb[i]])/2.),
                         color=cmap(norm(tmp_data[i])),zorder=i)  
        # add lines around each tile
        plt.plot((ll[i]+np.array([-dll[i],+dll[i],+dll[i],-dll[i],-dll[i]])/2.),
                 (bb[i]+np.array([-dbb[i],-dbb[i],+dbb[i],+dbb[i],-dbb[i]])/2.),
                 color='black',linewidth=1,zorder=1000+i)

    # add standard labels
    plt.xlabel('Azimuth')
    plt.ylabel('Zenith')

    # use these limits
    #plt.xlim(-np.pi,np.pi)
    #plt.ylim(0,np.pi)
    # not
    

def zero_func(x,y,grid=False):
    """Returns an array of zeros for any given input array (or scalar) x.
    :param: x       Input array (or scalar, tuple, list)
    :param: y       Optional second array or value
    :option: grid   Standard keyword to work with RectBivarSpline
    """
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.zeros(len(x))
    else:
        return 0.
    
    
def get_response(Response,zenith,azimuth,deg=False,cut=65): 
    """
    Calculate response for a given zenith/azimuth position up to a certain threshold cut in zenith.
    :param: Response    List(!) of response function to be looped over in 1D
    :param: zenith      Zenith position of source with respect to instrument (in rad)
    :param: azimuth     Azimuth position of source with respect to instrument (in rad)
    :option: deg        Default False, if True, degree angles are converted into radians
    :option: cut        Default 65 deg, range up to which response is calculated (COSI FoV ~ 60 deg)
    """
    # check for rad input
    if deg == True:
        zenith, azimuth = np.deg2rad(zenith), np.deg2rad(azimuth)
    # loop over response list
    val = np.array([Response[r](zenith,azimuth,grid=False) for r in range(len(Response))])
    # check that values are strictly non-negative (may have happened during RectBivarSpline interpolation)
    val[val<0] = 0.
    # zero out values beyond the cut (and if negative zeniths happened)
    val[:,(zenith < 0) | (zenith > np.deg2rad(cut))] = 0
    # make everything finite
    val[np.isnan(val)] = 0.
    
    return val


def get_response_with_weights(Response,zenith,azimuth,deg=True,binsize=5,cut=60.0):
    """
    Calculate response at given zenith/azimuth position of a source relative to COSI,
    using the angular distance to the 4 neighbouring pixels that overlap using a 
    certain binsize.
    Note that this also introduces some smoothing of the response as zeniths/azimuths
    on the edges or corners of pixels will not be weighted equally but still get 
    contributions from the remaining pixels.
    :param: Response      Response grid with regular sky pixel dimension (zenith x azimuth)
                          and unfolded 1D phi-psi-chi dimension.
    :param: zenith        Zenith positions of the source with respect to the instrument (in deg)
    :param: azimuth       Azimuth positions of the source with respect to the instrument (in deg)
    :option: deg          Default True, (right now not checked for any purpose)
    :option: binsize      Default 5 deg (matching the sky dimension of the response). If set
                          differently, make sure it matches the sky dimension as otherwise,
                          false results may be returned
    :option: cut          Threshold to cut the response calculation after a certain zenith angle.
                          Default 60 deg (~ COSI FoV)
    
    Returns an array of length equal the response that is input.
    """
    # calculate the weighting for neighbouring pixels using their angular distance
    # also returns the indices of which pixels to be used for response averaging
    widx = get_response_weights_vector(zenith,azimuth,binsize,cut=cut)
    # This is a vectorised function so that each entry gets its own weighting
    # at the correct positions of the input angles ([:, None] is the same as 
    # column-vector multiplcation of a lot of ones)
    rsp0 = Response[widx[0][0,1,:],widx[0][0,0,:],:]*widx[1][0,:][:, None]
    rsp1 = Response[widx[0][1,1,:],widx[0][1,0,:],:]*widx[1][1,:][:, None]
    rsp2 = Response[widx[0][2,1,:],widx[0][2,0,:],:]*widx[1][2,:][:, None]
    rsp3 = Response[widx[0][3,1,:],widx[0][3,0,:],:]*widx[1][3,:][:, None]
    # add together
    rsp_mean = rsp0 + rsp1 + rsp2 + rsp3

    return rsp_mean


def get_response_weights_vector(zenith,azimuth,binsize=5,cut=57.4):
    """
    Get Compton response pixel weights (four nearest neighbours),
    weighted by angular distance to zenith/azimuth vector(!) input.
    Binsize determines regular(!!!) sky coordinate grid in degrees.

    For single zenith/azimuth pairs use get_response_weights()
    
    :param: zenith        Zenith positions of the source with respect to the instrument (in deg)
    :param: azimuth       Azimuth positions of the source with respect to the instrument (in deg)
    :option: binsize      Default 5 deg (matching the sky dimension of the response). If set
                          differently, make sure it matches the sky dimension as otherwise,
                          false results may be returned
    :option: cut          Threshold to cut the response calculation after a certain zenith angle.
                          Default 57.4 deg (0.1 deg before last pixel reaching beyon 60 deg)
    """

    # assuming useful input:
    # azimuthal angle is periodic in the range [0,360[
    # zenith ranges from [0,180[ 

    # check which pixel (index) was hit on regular grid
    hit_pixel_zi = np.floor(zenith/binsize)
    hit_pixel_ai = np.floor(azimuth/binsize)

    # and which pixel centre
    hit_pixel_z = (hit_pixel_zi+0.5)*binsize
    hit_pixel_a = (hit_pixel_ai+0.5)*binsize

    # check which zeniths are beyond threshold
    bad_idx = np.where(hit_pixel_z > cut) 
    
    # calculate nearest neighbour pixels indices
    za_idx = np.array([[np.floor(azimuth/binsize+0.5),np.floor(zenith/binsize+0.5)],
                       [np.floor(azimuth/binsize+0.5),np.floor(zenith/binsize-0.5)],
                       [np.floor(azimuth/binsize-0.5),np.floor(zenith/binsize+0.5)],
                       [np.floor(azimuth/binsize-0.5),np.floor(zenith/binsize-0.5)]]).astype(int)

    # take care of bounds at zenith (azimuth is allowed to be -1!)
    (za_idx[:,1,:])[np.where(za_idx[:,1,:] < 0)] += 1
    (za_idx[:,1,:])[np.where(za_idx[:,1,:] >= 180/binsize)] = int(180/binsize-1)
    # but azimuth may not be larger than range [0,360/binsize[
    (za_idx[:,0,:])[np.where(za_idx[:,0,:] >= 360/binsize)] = 0
    
    # and pixel centres of neighbours
    azimuth_neighbours = (za_idx[:,0]+0.5)*binsize
    zenith_neighbours = (za_idx[:,1]+0.5)*binsize

    # calculate angular distances to neighbours
    dists = angular_distance(azimuth_neighbours,zenith_neighbours,azimuth,zenith)

    # inverse weighting to get impact of neighbouring pixels
    n_in = len(zenith)
    weights = (1/dists)/np.sum(1/dists,axis=0).repeat(4).reshape(n_in,4).T
    # if pixel is hit directly, set weight to 1.0
    weights[np.isnan(weights)] = 1
    # set beyond threshold weights to zero
    weights[:,bad_idx] = 0

    return za_idx,weights


def get_response_weights(zenith,azimuth,binsize=5,verbose=False,cut=57.4):
    """
    Get Compton response pixel weights (four nearest neighbours),
    weighted by angular distance to zenith/azimuth input.
    Binsize determines regular(!!!) sky coordinate grid in degrees.

    TS: This is the version to calculate a single zenith/azimuth pair response
    as would be needed if there was a stationary (staring) instrument
    This is then faster because the response only needs to be calculated once.
    The rest is the same as get_response_weights_vector().
    
    :param: zenith        Zenith positions of the source with respect to the instrument (in deg)
    :param: azimuth       Azimuth positions of the source with respect to the instrument (in deg)
    :option: binsize      Default 5 deg (matching the sky dimension of the response). If set
                          differently, make sure it matches the sky dimension as otherwise,
                          false results may be returned
    :option: cut          Threshold to cut the response calculation after a certain zenith angle.
                          Default 57.4 deg (0.1 deg before last pixel reaching beyon 60 deg)    
    """

    # check input zenith and azimuth to be in a reasonable range
    # azimuthal angle is periodic, so add/subtract 360 until it
    # is in the range [0,360[
    while (azimuth < 0) | (azimuth >= 360):
        if azimuth < 0:
            azimuth += 360
        if azimuth > 360:
            azimuth -= 360

    # zenith ranges from [0,180[ so that out of bounds angles
    # will be set to bounds (should not happen in normal cases)
    if zenith < 0:
        zenith = 0
    if zenith > 180:
        zenith = 180

    # check which pixel (index) was hit on regular grid
    hit_pixel_zi = np.floor(zenith/binsize)
    hit_pixel_ai = np.floor(azimuth/binsize)
    verb(verbose,(hit_pixel_ai,hit_pixel_zi))
    
    # and which pixel centre
    hit_pixel_z = (hit_pixel_zi+0.5)*binsize
    hit_pixel_a = (hit_pixel_ai+0.5)*binsize
    verb(verbose,(hit_pixel_a,hit_pixel_z))

    # check for threshold:
    if hit_pixel_z > cut:
        return np.zeros((4,2),dtype=int),np.zeros(4)
    
    # calculate nearest neighbour pixels indices
    za_idx = np.array([[np.floor(azimuth/binsize+0.5),np.floor(zenith/binsize+0.5)],
                       [np.floor(azimuth/binsize+0.5),np.floor(zenith/binsize-0.5)],
                       [np.floor(azimuth/binsize-0.5),np.floor(zenith/binsize+0.5)],
                       [np.floor(azimuth/binsize-0.5),np.floor(zenith/binsize-0.5)]]).astype(int)
    verb(verbose,za_idx)
         
    # take care of bounds at zenith (azimuth is allowed to be -1!)
    za_idx[np.where(za_idx[:,1] < 0),1] += 1
    za_idx[np.where(za_idx[:,0] >= 360/binsize),0] = 0
    verb(verbose,za_idx)
         
    # and pixel centres of neighbours
    azimuth_neighbours = (za_idx[:,0]+0.5)*binsize
    zenith_neighbours = (za_idx[:,1]+0.5)*binsize
    verb(verbose,(azimuth_neighbours,zenith_neighbours))
         
    # calculate angular distances to neighbours
    dists = angular_distance(azimuth_neighbours,zenith_neighbours,azimuth,zenith)
    verb(verbose,dists)
    
    # inverse weighting to get impact of neighbouring pixels
    weights = (1/dists)/np.sum(1/dists)
    # if pixel is hit directly, set weight to 1.0
    weights[np.isnan(weights)] = 1

    return za_idx.astype(int),weights
  
    
def angular_distance(l1,b1,l2,b2,deg=True):
    """
    Calculate angular distance on a sphere from longitude/latitude pairs to other using Great circles
    in units of deg
    :param: l1    longitude of point 1 (or several)
    :param: b1    latitude of point 1 (or several)
    :param: l2    longitude of point 2
    :param: b2    latitude of point 2
    :option: deg  option carried over to GreatCricle routine
    """
    # calculate the Great Circle between the two points
    # this is a geodesic on a sphere and describes the shortest distance
    gc = GreatCircle(l1,b1,l2,b2,deg=deg)
    
    # check for exceptions
    if gc.size == 1:
        if gc > 1:
            gc = 1.
    else:
        gc[np.where(gc > 1)] = 1.

    return np.rad2deg(np.arccos(gc))    
    
    
def GreatCircle(l1,b1,l2,b2,deg=True):
    """
    Calculate the Great Circle length on a sphere from longitude/latitude pairs to others
    in units of rad on a unit sphere
    :param: l1    longitude of point 1 (or several)
    :param: b1    latitude of point 1 (or several)
    :param: l2    longitude of point 2
    :param: b2    latitude of point 2
    :option: deg  Default True to convert degree input to radians for trigonometric function use
                  If False, radian input is assumed
    """
    if deg == True:
        l1,b1,l2,b2 = np.deg2rad(l1),np.deg2rad(b1),np.deg2rad(l2),np.deg2rad(b2)

    return np.sin(b1)*np.sin(b2) + np.cos(b1)*np.cos(b2)*np.cos(l1-l2)    
    
    
def get_response_with_weights_linear(Response,zenith,azimuth,deg=True,binsize=5,cut=60.0):
    """
    Calculate response at given zenith/azimuth position of a source relative to COSI,
    using the EUCLIDEAN distance to the 4 neighbouring pixels that overlap using a 
    certain binsize.
    Note that this also introduces some smoothing of the response as zeniths/azimuths
    on the edges or corners of pixels will not be weighted equally but still get 
    contributions from the remaining pixels.
    :param: Response      Response grid with regular sky pixel dimension (zenith x azimuth)
                          and unfolded 1D phi-psi-chi dimension.
    :param: zenith        Zenith positions of the source with respect to the instrument (in deg)
    :param: azimuth       Azimuth positions of the source with respect to the instrument (in deg)
    :option: deg          Default True, (right now not checked for any purpose)
    :option: binsize      Default 5 deg (matching the sky dimension of the response). If set
                          differently, make sure it matches the sky dimension as otherwise,
                          false results may be returned
    :option: cut          Threshold to cut the response calculation after a certain zenith angle.
                          Default 60 deg (~ COSI FoV)
    
    Returns an array of length equal the response that is input.
    """
    # calculate the weighting for neighbouring pixels using their EUCLIDEAN distance
    # also returns the indices of which pixels to be used for response averaging
    widx = get_response_weights_vector_linear(zenith,azimuth,binsize,cut=cut)
    # This is a vectorised function so that each entry gets its own weighting
    # at the correct positions of the input angles ([:, None] is the same as 
    # column-vector multiplcation of a lot of ones)
    rsp0 = Response[widx[0][0,1,:],widx[0][0,0,:],:]*widx[1][0,:][:, None]
    rsp1 = Response[widx[0][1,1,:],widx[0][1,0,:],:]*widx[1][1,:][:, None]
    rsp2 = Response[widx[0][2,1,:],widx[0][2,0,:],:]*widx[1][2,:][:, None]
    rsp3 = Response[widx[0][3,1,:],widx[0][3,0,:],:]*widx[1][3,:][:, None]
    # add together
    rsp_mean = rsp0 + rsp1 + rsp2 + rsp3

    return rsp_mean    
    
    
def euclidean_distance(x1,y1,x2,y2):
    """
    Returns the Euclidean distance (L2 norm) of a pair of coordinates.
    :param: x1      x coordinate of point 1 (also several)
    :param: y1      y coordinate of point 1 (also several)
    :param: x2      x coordinate of point 2
    :param: y2      y coordinate of point 2
    """
    return np.sqrt((x1-x2)**2+(y1-y2)**2)

def get_response_weights_vector_linear(zenith,azimuth,binsize=5,cut=57.4):
    """
    Get Compton response pixel weights (four nearest neighbours),
    weighted by EUCLIDEAN distance to zenith/azimuth vector(!) input.
    Binsize determines regular(!!!) sky coordinate grid in degrees.

    For single zenith/azimuth pairs use get_response_weights()
    
    :param: zenith        Zenith positions of the source with respect to the instrument (in deg)
    :param: azimuth       Azimuth positions of the source with respect to the instrument (in deg)
    :option: binsize      Default 5 deg (matching the sky dimension of the response). If set
                          differently, make sure it matches the sky dimension as otherwise,
                          false results may be returned
    :option: cut          Threshold to cut the response calculation after a certain zenith angle.
                          Default 57.4 deg (0.1 deg before last pixel reaching beyon 60 deg)
    """

    # assuming useful input:
    # azimuthal angle is periodic in the range [0,360[
    # zenith ranges from [0,180[ 

    # check which pixel (index) was hit on regular grid
    hit_pixel_zi = np.floor(zenith/binsize)
    hit_pixel_ai = np.floor(azimuth/binsize)

    # and which pixel centre
    hit_pixel_z = (hit_pixel_zi+0.5)*binsize
    hit_pixel_a = (hit_pixel_ai+0.5)*binsize

    # check which zeniths are beyond threshold
    bad_idx = np.where(hit_pixel_z > cut) 
    
    # calculate nearest neighbour pixels indices
    za_idx = np.array([[np.floor(azimuth/binsize+0.5),np.floor(zenith/binsize+0.5)],
                       [np.floor(azimuth/binsize+0.5),np.floor(zenith/binsize-0.5)],
                       [np.floor(azimuth/binsize-0.5),np.floor(zenith/binsize+0.5)],
                       [np.floor(azimuth/binsize-0.5),np.floor(zenith/binsize-0.5)]]).astype(int)

    # take care of bounds at zenith (azimuth is allowed to be -1!)
    (za_idx[:,1,:])[np.where(za_idx[:,1,:] < 0)] += 1
    (za_idx[:,1,:])[np.where(za_idx[:,1,:] >= 180/binsize)] = int(180/binsize-1)
    # but azimuth may not be larger than range [0,360/binsize[
    (za_idx[:,0,:])[np.where(za_idx[:,0,:] >= 360/binsize)] = 0
    
    # and pixel centres of neighbours
    azimuth_neighbours = (za_idx[:,0]+0.5)*binsize
    zenith_neighbours = (za_idx[:,1]+0.5)*binsize

    # calculate EUCLIDEAN distances to neighbours
    dists = euclidean_distance(azimuth_neighbours,zenith_neighbours,azimuth,zenith)

    # inverse weighting to get impact of neighbouring pixels
    n_in = len(zenith)
    weights = (1/dists)/np.sum(1/dists,axis=0).repeat(4).reshape(n_in,4).T
    # if pixel is hit directly, set weight to 1.0
    weights[np.isnan(weights)] = 1
    # set beyond threshold weights to zero
    weights[:,bad_idx] = 0

    return za_idx,weights    
    

def get_response_with_weights_area(Response,zenith,azimuth,deg=True,binsize=5,cut=60.0):
    """
    Calculate response at given zenith/azimuth position of a source relative to COSI,
    using the overlapping areas of the 4 neighbouring pixels that overlap using a 
    certain binsize.
    !!! Note that this is NOT WORKING PROPERLY right now because the pixels are not   !!!
    !!! assigned in the correct way all the time which is seen in searching for point !!!
    !!! sources and the l/b correlation plot. Need to check later (oom is still okay) !!!
    :param: Response      Response grid with regular sky pixel dimension (zenith x azimuth)
                          and unfolded 1D phi-psi-chi dimension.
    :param: zenith        Zenith positions of the source with respect to the instrument (in deg)
    :param: azimuth       Azimuth positions of the source with respect to the instrument (in deg)
    :option: deg          Default True, (right now not checked for any purpose)
    :option: binsize      Default 5 deg (matching the sky dimension of the response). If set
                          differently, make sure it matches the sky dimension as otherwise,
                          false results may be returned
    :option: cut          Threshold to cut the response calculation after a certain zenith angle.
                          Default 60 deg (~ COSI FoV)
    
    Returns an array of length equal the response that is input.
    """
    # calculate the weighting for neighbouring pixels using their overlapping area
    # also returns the indices of which pixels to be used for response averaging
    widx = get_response_weights_vector_area(zenith,azimuth,binsize,cut=60)
    # This is a vectorised function so that each entry gets its own weighting
    # at the correct positions of the input angles ([:, None] is the same as 
    # column-vector multiplcation of a lot of ones)
    rsp0 = Response[widx[0][0,1,:],widx[0][0,0,:],:]*widx[1][0,:][:, None]
    rsp1 = Response[widx[0][1,1,:],widx[0][1,0,:],:]*widx[1][1,:][:, None]
    rsp2 = Response[widx[0][2,1,:],widx[0][2,0,:],:]*widx[1][2,:][:, None]
    rsp3 = Response[widx[0][3,1,:],widx[0][3,0,:],:]*widx[1][3,:][:, None]
    # add together
    rsp_mean = rsp0 + rsp1 + rsp2 + rsp3

    return rsp_mean    

    
def get_response_weights_vector_area(zenith,azimuth,binsize=5,cut=57.4):
    """
    Get Compton response pixel weights (four nearest neighbours),
    weighted by overlapping pixel areas for each zenith/azimuth vector(!) input.
    Binsize determines regular(!!!) sky coordinate grid in degrees.

    For single zenith/azimuth pairs use get_response_weights_area()
    
    :param: zenith        Zenith positions of the source with respect to the instrument (in deg)
    :param: azimuth       Azimuth positions of the source with respect to the instrument (in deg)
    :option: binsize      Default 5 deg (matching the sky dimension of the response). If set
                          differently, make sure it matches the sky dimension as otherwise,
                          false results may be returned
    :option: cut          Threshold to cut the response calculation after a certain zenith angle.
                          Default 57.4 deg (0.1 deg before last pixel reaching beyon 60 deg)
    """

    # assuming useful input:
    # azimuthal angle is periodic in the range [0,360[
    # zenith ranges from [0,180[ 

    # check which pixel (index) was hit on regular grid
    hit_pixel_zi = np.floor(zenith/binsize)
    hit_pixel_ai = np.floor(azimuth/binsize)

    # and which pixel centre
    hit_pixel_z = (hit_pixel_zi+0.5)*binsize
    hit_pixel_a = (hit_pixel_ai+0.5)*binsize

    # check which zeniths are beyond threshold
    bad_idx = np.where(hit_pixel_z > cut) 
    
    # calculate nearest neighbour pixels indices
    za_idx = np.array([[np.floor(azimuth/binsize+0.5),np.floor(zenith/binsize+0.5)],
                       [np.floor(azimuth/binsize+0.5),np.floor(zenith/binsize-0.5)],
                       [np.floor(azimuth/binsize-0.5),np.floor(zenith/binsize+0.5)],
                       [np.floor(azimuth/binsize-0.5),np.floor(zenith/binsize-0.5)]]).astype(int)

    # calculate the areas of overlap (always the same)
    # only which value is assigned to which pixel changes (see distance below)
    DeltaZ = np.abs(zenith-hit_pixel_z)
    DeltaA = np.abs(azimuth-hit_pixel_a)
    R = (binsize**2 - binsize*(DeltaZ+DeltaA) + DeltaZ*DeltaA)/binsize**2
    S = (binsize*DeltaZ - DeltaZ*DeltaA)/binsize**2
    P = (binsize*DeltaA - DeltaZ*DeltaA)/binsize**2
    Q = (DeltaZ*DeltaA)/binsize**2
    
    # take care of bounds at zenith (azimuth is allowed to be -1!)
    (za_idx[:,1,:])[np.where(za_idx[:,1,:] < 0)] += 1
    (za_idx[:,1,:])[np.where(za_idx[:,1,:] >= 180/binsize)] = int(180/binsize-1)
    # but azimuth may not be larger than range [0,360/binsize[
    (za_idx[:,0,:])[np.where(za_idx[:,0,:] >= 360/binsize)] = 0
    
    # and pixel centres of neighbours
    azimuth_neighbours = (za_idx[:,0]+0.5)*binsize
    zenith_neighbours = (za_idx[:,1]+0.5)*binsize

    # calculate angular distances to neighbours
    # this is done to get which pixel should get which weighting for the overlap area
    dists = euclidean_distance(azimuth_neighbours,zenith_neighbours,azimuth,zenith)

    # inverse weighting to get impact of neighbouring pixels
    n_in = len(zenith)
    dweights = (1/dists)/np.sum(1/dists,axis=0).repeat(4).reshape(n_in,4).T
    # if pixel is hit directly, set weight to 1.0
    dweights[np.isnan(dweights)] = 1
    # set beyond threshold weights to zero
    dweights[:,bad_idx] = 0

    weights = np.array([R,S,P,Q])
#    print('weights: ',weights.shape)
#    print('dweights: ',dweights.shape)
#    print('dists: ',dists.shape)

#    print(np.sort(weights,axis=0))
#    print(np.argsort(dweights,axis=0))
    
    # probably, this is not correctly working. Don't know how to assign which weighting to which pixel
    # in a fast way
    weights = np.sort(weights,axis=0)[np.flip(np.argsort(dweights,axis=0))][:,0,:]
    
    return za_idx,weights

    
def get_response_weights_area(zenith,azimuth,binsize=5,verbose=False,cut=57.4):
    """
    Get Compton response pixel weights (four nearest neighbours),
    weighted by overlapping pixel areas for each zenith/azimuth vector(!) input.
    Binsize determines regular(!!!) sky coordinate grid in degrees.

    TS: This is the version to calculate a single zenith/azimuth pair response
    as would be needed if there was a stationary (staring) instrument
    This is then faster because the response only needs to be calculated once.
    The rest is the same as get_response_weights_area_vector().
    
    :param: zenith        Zenith positions of the source with respect to the instrument (in deg)
    :param: azimuth       Azimuth positions of the source with respect to the instrument (in deg)
    :option: binsize      Default 5 deg (matching the sky dimension of the response). If set
                          differently, make sure it matches the sky dimension as otherwise,
                          false results may be returned
    :option: cut          Threshold to cut the response calculation after a certain zenith angle.
                          Default 57.4 deg (0.1 deg before last pixel reaching beyon 60 deg)    
    """

    # check input zenith and azimuth to be in a reasonable range
    # azimuthal angle is periodic, so add/subtract 360 until it
    # is in the range [0,360[
    while (azimuth < 0) | (azimuth >= 360):
        if azimuth < 0:
            azimuth += 360
        if azimuth > 360:
            azimuth -= 360

    # zenith ranges from [0,180[ so that out of bounds angles
    # will be set to bounds (should not happen in normal cases)
    if zenith < 0:
        zenith = 0
    if zenith > 180:
        zenith = 180

    # check which pixel (index) was hit on regular grid
    hit_pixel_zi = np.floor(zenith/binsize)
    hit_pixel_ai = np.floor(azimuth/binsize)
    verb(verbose,(hit_pixel_ai,hit_pixel_zi))
    
    # and which pixel centre
    hit_pixel_z = (hit_pixel_zi+0.5)*binsize
    hit_pixel_a = (hit_pixel_ai+0.5)*binsize
    verb(verbose,(hit_pixel_a,hit_pixel_z))

    # check for threshold:
    if hit_pixel_z > cut:
        return np.zeros((4,2),dtype=int),np.zeros(4)
    
    # calculate nearest neighbour pixels indices
    za_idx = np.array([[np.floor(azimuth/binsize+0.5),np.floor(zenith/binsize+0.5)],
                       [np.floor(azimuth/binsize+0.5),np.floor(zenith/binsize-0.5)],
                       [np.floor(azimuth/binsize-0.5),np.floor(zenith/binsize+0.5)],
                       [np.floor(azimuth/binsize-0.5),np.floor(zenith/binsize-0.5)]]).astype(int)
    verb(verbose,za_idx)

    DeltaZ = np.abs(zenith-hit_pixel_z)
    DeltaA = np.abs(azimuth-hit_pixel_a)
    R = (binsize**2 - binsize*(DeltaZ+DeltaA) + DeltaZ*DeltaA)/binsize**2
    S = (binsize*DeltaZ - DeltaZ*DeltaA)/binsize**2
    P = (binsize*DeltaA - DeltaZ*DeltaA)/binsize**2
    Q = (DeltaZ*DeltaA)/binsize**2

    verb(verbose,[R,S,P,Q])
    
    # take care of bounds at zenith (azimuth is allowed to be -1!)
    za_idx[np.where(za_idx[:,1] < 0),1] += 1
    za_idx[np.where(za_idx[:,0] >= 360/binsize),0] = 0
    verb(verbose,za_idx)
         
    # and pixel centres of neighbours
    azimuth_neighbours = (za_idx[:,0]+0.5)*binsize
    zenith_neighbours = (za_idx[:,1]+0.5)*binsize
    verb(verbose,(azimuth_neighbours,zenith_neighbours))
         
    # calculate angular distances to neighbours
    dists = euclidean_distance(azimuth_neighbours,zenith_neighbours,azimuth,zenith)
    verb(verbose,dists)
    
    # inverse weighting to get impact of neighbouring pixels
    dweights = (1/dists)/np.sum(1/dists)
    # if pixel is hit directly, set weight to 1.0
    dweights[np.isnan(dweights)] = 1

    weights = np.array([R,S,P,Q])

    weights = (np.sort(weights))[np.flip(np.argsort(dweights))]

    return za_idx.astype(int),weights    
    

def construct_sc_matrix(scx_l, scx_b, scy_l, scy_b, scz_l, scz_b):
    """
    Construct orientation matrix for your spacecraft/intrument to rotate later into different frames
    :param: scx_l      longitude of x-direction/coordinate
    :param: scx_b      latitude of x-direction/coordinate
    :param: scy_l      longitude of y-direction/coordinate
    :param: scy_b      latitude of y-direction/coordinate
    :param: scz_l      longitude of z-direction/coordinate
    :param: scz_b      latitude of z-direction/coordinate
    """
    # init matrix
    sc_matrix = np.zeros((3,3))
    # calculate coordinates
    sc_matrix[0,:] = polar2cart(scx_l, scx_b)
    sc_matrix[1,:] = polar2cart(scy_l, scy_b)
    sc_matrix[2,:] = polar2cart(scz_l, scz_b)
    
    return sc_matrix


def zenazi(scx_l, scx_b, scy_l, scy_b, scz_l, scz_b, src_l, src_b):
    """
    # from spimodfit zenazi function (with rotated axes (optical axis for COSI = z)
    # calculate angular distance wrt optical axis in zenith (theta) and
    # azimuth (phi): (zenazi function)
    # input: spacecraft pointing directions sc(xyz)_l/b; source coordinates src_l/b
    # output: source coordinates in spacecraft system frame
    
    Calculate zenith and azimuth angle of a point (a source) given the orientations
    of an instrument (or similar) in a certain coordinate frame (e.g. galactic).
    Each point in galactic coordinates can be uniquely mapped into zenith/azimuth of
    an instrument/observer/..., by using three Great Circles in x/y/z and retrieving
    the correct angles
    
    :param: scx_l      longitude of x-direction/coordinate
    :param: scx_b      latitude of x-direction/coordinate
    :param: scy_l      longitude of y-direction/coordinate
    :param: scy_b      latitude of y-direction/coordinate
    :param: scz_l      longitude of z-direction/coordinate
    :param: scz_b      latitude of z-direction/coordinate
    :param: src_l      SOURCE longitude
    :param: src_b      SOURCE latitude
    
    Space craft coordinates can also be vectors for quick computation fo arrays
    """
    # I cannot describe this is words, please google zenith/azimuth and find a picture
    # where you understand which angles are being calculated
    # Zenith is the distance from the optical axis (here z)
    costheta = GreatCircle(scz_l,scz_b,src_l,src_b)                                                                        
    # Azimuth is the combination of the remaining two
    cosx = GreatCircle(scx_l,scx_b,src_l,src_b)
    cosy = GreatCircle(scy_l,scy_b,src_l,src_b)
    
    # check exceptions
    # maybe not for vectorisation
    """
    if costheta.size == 1:
        if (costheta > 1.0):
            costheta = 1.0
        if (costheta < -1.0):
            costheta = -1.0
    else:
        costheta[costheta > 1.0] = 1.0
        costheta[costheta < -1.0] = -1.0
    """
    # theta = zenith
    theta = np.rad2deg(np.arccos(costheta))
    # phi = azimuth
    phi = np.rad2deg(np.arctan2(cosx,cosy))
    
    # make azimuth going from 0 to 360 deg
    if phi.size == 1:
        if (phi < 0):
            phi += 360
    else:
        phi[phi < 0] += 360
    
    return theta,phi    
    
    
def lima_significance(Non,Noff,alpha):
    """
    Calculate the significance of a source with background using an off measurement (or background simulation)
    using the likelihood ratio test including Poisson count statistics. From Li & Ma (1983). Please, also have
    a look at Vianello (2018) for extended versions of this combining Gaussian and Poisson likelihood, for exaple.
    
    :param: Non     Counts in the "on" measurement (SRC plus BG)
    :param: Noff    Counts in the "off" measurement (BG only, or simulation with amplitude estimate)
    :param: alpha   Ratio between observation times in on and off measurement
    """
    return np.sqrt(2* (Non * np.log((1+alpha)/alpha*(Non/(Non+Noff))) + Noff*np.log((1+alpha)*Noff/(Non+Noff))) )
    
    
def vector_length(x,y,z):
    """
    Return Euclidean (L2) norm of a 3D vector (or series of vectors in x/y/z coordiantes):
    
    :param: x    x value(s)
    :param: y    y value(s)
    :param: z    z value(s)
    """
    return np.sqrt(x**2+y**2+z**2)
    
    
def get_pixel_weights(COSI_Data,tdx,day,hour,pixel_size=1,plot=False):
    """
    Return populated sky pixels in l/b coordinates, where COSI
    looked at a certain day and hour with given square pixel size
    (default 1x1 deg2).
    Also return the weights at each pixel (i.e. how many events
    have been recorded during that time).
    """
    # get l and b pointing positions in a certain hour, convert to deg
    l_hour = COSI_Data[day]['Zpointings'][tdx[day][hour]['Indices'],0]
    b_hour = COSI_Data[day]['Zpointings'][tdx[day][hour]['Indices'],1]
    l_hour = np.rad2deg(l_hour).reshape(l_hour.size)
    l_hour[l_hour > 180] -= 360
    b_hour = np.rad2deg(b_hour).reshape(b_hour.size)

    # define min and max of pixel grid
    lmin,lmax = np.floor(l_hour.min()),np.ceil(l_hour.max())
    bmin,bmax = np.floor(b_hour.min()),np.ceil(b_hour.max())

    # make 2D histogram with given pixel size (all-sky)
    l_grid = np.linspace(-180,180,np.round(360/pixel_size).astype(int)+1)
    b_grid = np.linspace(-90,90,np.round(180/pixel_size).astype(int)+1)
    weights = np.histogram2d(l_hour,b_hour,bins=(l_grid,b_grid))

    if plot == True:
        _ = plt.hist2d(l_hour,b_hour,bins=(l_grid,b_grid),cmap=plt.cm.Blues)
        for bi in range(np.abs(bmax-bmin).astype(int)):
            for li in range(np.abs(lmax-lmin).astype(int)):
                text = weights[0][li+180+lmin.astype(int),bi+90+bmin.astype(int)]
                if text != 0:
                    plt.text(weights[1][li+180]+0.5+lmin.astype(int),
                             weights[2][bi+90]+0.5+bmin.astype(int),
                             text.astype(int),
                             horizontalalignment='center',
                             verticalalignment='center',
                             color='xkcd:yellowish orange',
                             fontsize=16)
        plt.xlim(minmax(l_hour)+[-1,1])
        plt.ylim(minmax(b_hour)+[-1,1])
        plt.xlabel('Gal. lon. [deg]')
        plt.ylabel('Gal. lat. [deg]')
        plt.title('Weights in %3.2f x %3.2f deg pixels' % (pixel_size,pixel_size),fontsize=24)
    
    return weights    
    
    
def circle_on_the_sky(ls,bs,th,n_points=100):
    """
    Returns (galactic) coordinates of a circle with with radius th
    with its centre at a position on a sphere (ls/bs).
    Default are n_points=100 points.
    All angles are to be given in degree!
    
    :param: ls          longitude centre of the circle (deg)
    :param: bs          latitude centre of the circle (deg)
    :param: th          radius of circle on the sky (deg)
    :option: n_points   Default 100, number of points to calculate
    """
    from scipy.spatial.transform import Rotation as R

    thr = np.deg2rad(th)

    # start from the circle centre point at galactic coordiantes 0/0 on that sphere
    # TS: the difficult thing is just to figure out what angle corresponds to what axis
    vec = np.array([np.cos(thr),0,0])
    # rotate that point to the wanted position
    r = R.from_euler('yz',[bs+180,ls+180],degrees=True)
    rot_vec = r.apply(vec)
    # initial and rotated point are NOT UNIT VECTORS, thus normalise when required

    # get points of that circle (radius sin(th), AT position cos(th))
    alpha = np.linspace(-np.pi,np.pi,n_points)
    circle_vec = np.array([np.ones(len(alpha))*np.cos(thr),
                           np.sin(thr)*np.cos(alpha),
                           np.sin(thr)*np.sin(alpha)])
    # rotate these points in the same way
    rot_circle_vec = []
    for i in range(len(alpha)):
        rot_circle_vec.append(r.apply(circle_vec[:,i]))
    rot_circle_vec = np.array(rot_circle_vec).T
    # should not happen, but let's make sure
    rot_circle_vec[2,rot_circle_vec[2,:] < -1] = -1
    
    # calculate l and b coordiantes from (cartesian to spherical on unit sphere)
    b_calc = np.rad2deg(np.arcsin(rot_circle_vec[2,:]/
                                  vector_length(rot_circle_vec[0,:],
                                                rot_circle_vec[1,:],
                                                rot_circle_vec[2,:])))
    l_calc = np.rad2deg(np.arctan2(rot_circle_vec[1,:],rot_circle_vec[0,:]))
    
    return l_calc,b_calc    


def tsgt():
    """
    Return 0
    """
    return 0


def hallo():
    """
    German "HELLO WORLD! :) ;) ~.~ ^^ O_o :p :b d: q:" example
    """
    print('hallo')


def word_replace(in_file,out_file,checkWords,repWords):
    """
    Used to replace multiple strings in a file and write a new file
    param: in_file: input file in read-only mode
    param: out_file: duplicated file, line by line for which
    param: checkWords (array of strings) are to be replaced with
    param: repWords (array of strings)
    """ 
#checkWords = ('1467094373','OP_160628_1hour.ori')
#repWords = (str(last_time_ceiled),'1hour.ori'+str('%02i' % i))

    f1 = open(in_file, 'r')
    f2 = open(out_file, 'w')
    for line in f1:
        for check, rep in zip(checkWords, repWords):
            line = line.replace(check, rep)
        f2.write(line)
    f1.close()
    f2.close()
