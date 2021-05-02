#!/usr/bin/env python

# ==============================================================================
# COPERNICUS POSTPROCESSING UTILITIES MODULE
# Tools related to postprocessing NetCDF data from 35GHz Chilbolton Radar Copernicus
#
# Author:        Chris Walden, NCAS
# History:
# Version:	 0.2
# Last modified: 22/10/20
# ==============================================================================
module_version = 0.2

# ------------------------------------------------------------------------------
# Import required tools
# ------------------------------------------------------------------------------
import numpy as np
import numpy.ma as ma;
import os, re, sys, getopt, shutil, zipfile, string, pwd, getpass
import netCDF4 as nc4
import socket

import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt  #import plotting package
import pyart


from datetime import tzinfo, datetime, time

# ------------------------------------------------------------------------------
# Define useful function
# ------------------------------------------------------------------------------
def in_interval(seq,xmin,xmax):
"""Select part of sequence that lies in a given interval [xmin,xmax[ """
    for i, x in enumerate(seq):
        if x>=xmin and x<xmax:
            yield i

# ------------------------------------------------------------------------------
# Get history from NetCDF file
# ------------------------------------------------------------------------------
def get_history(ncfile):
"""Get history attribute from NetCDF file. """

    user = getpass.getuser()
    print(user)

    # ----------------
    # Open NetCDF file
    # ----------------
    print('Opening NetCDF file ' + ncfile)
    nc = nc4.Dataset(ncfile,'r+',format='NETCDF3_CLASSIC')

    print(nc.history)
    nc.close

# ------------------------------------------------------------------------------
# Update global attributes - institution and references
# ------------------------------------------------------------------------------
def update_metadata(ncfile):
"""Update metadata in NetCDF file to reflect processing carried out."""
    user = getpass.getuser()
    print(user)

    # ----------------
    # Open NetCDF file
    # ----------------
    print('Opening NetCDF file ' + ncfile)
    nc = nc4.Dataset(ncfile,'r+',format='NETCDF3_CLASSIC')

    institution  = ("National Centre for Atmospheric Science,"
                    + " UK: https://www.ncas.ac.uk\n")
    institution += ("Data held at the Centre for Environmental Data Analysis,"
                    + " UK: https://www.ceda.ac.uk")
    nc.institution = institution

    references = ""
    nc.references = references

    updttime = datetime.utcnow()
    updttimestr = updttime.ctime()

    history = updttimestr + (" - user:" + user
    + " machine: " + socket.gethostname()
    + " program: copernicus_utils.py update_metadata"
    + " version:" + str(module_version))

    nc.history += "\n" + history
    print(nc.history)
    nc.close

# ------------------------------------------------------------------------------
# Update global attributes - institution and references
# ------------------------------------------------------------------------------
def update_metadata_cfradial(cfradfile):
"""Update metadata in CFradial file."""
    user = getpass.getuser()
    print(user)

    # ----------------
    # Open NetCDF file
    # ----------------
    print('Opening NetCDF file ' + cfradfile)
    nc = nc4.Dataset(cfradfile,'r+',format='NETCDF4')

    institution  = ("National Centre for Atmospheric Science,"
                    + " UK: https://www.ncas.ac.uk\n")
    institution += ("Data held at the Centre for Environmental Data Analysis,"
                    + " UK: https://www.ceda.ac.uk")
    nc.institution = institution

    references = ""
    nc.references = references

    time0       = nc.variables['time'];
    time0.comment = ("Coordinate variable for time."
                + " Time at the centre of each ray in fractional seconds"
                + " since the global variable time_reference\n");

    range0      = nc.variables['range'];
    range0.units = "metres" ;
    range0.renameAttribute("standard_name","proposed_standard_name");
    range0.comment = ("Coordinate variable for range."
                + " Range to centre of each bin.");

    updttime = datetime.utcnow()
    updttimestr = updttime.ctime()

    history = updttimestr + (" - user:" + user
    + " machine: " + socket.gethostname()
    + " program: kepler_utils.py update_metadata_cfradial"
    + " version:" + str(module_version))

    nc.history += "\n" + history
    print(nc.history)
    nc.close



# ------------------------------------------------------------------------
# Define function to read mmclx files into pyart radar structure
# ------------------------------------------------------------------------
def read_copernicus(filename, **kwargs):
    """
    Read a netCDF file from Copernicus.

    Parameters
    ----------
    filename : str
        Name of netCDF file to read data from.

    Returns
    -------
    radar : Radar
        Radar object.

    """
    # test for non empty kwargs
    _test_arguments(kwargs)

    # create metadata retrieval object
    filemetadata = FileMetadata('copernicus')

    # read the data
    ncobj = netCDF4.Dataset(filename)
    ncvars = ncobj.variables

    # general parameters
    nrays = ncvars['azimuth'].shape[0]
    scan_type = 'fix'

    # time
    # interpolate between the first and last timestamps in the Time variable
    time = filemetadata('time')
    nctime = ncvars['time']
    z = [n for n in list(filename) if n.isdigit()]
    #z extracts the numbers from the name of file to a list
    year = ''.join(z[0:4])
    #join will create a number rather than a list object
    month = ''.join(z[4:6])
    day = ''.join(z[6:8])
    
    time['units'] = make_time_unit_str(
        datetime.datetime(int(year),int(month),int(day))) 
    #time['units'] = make_time_unit_str(
    #    datetime.datetime(int(filename[-21:-17]),int(filename[-17:-15]),int(filename[-15:-13]))) 
    time['data'] = nctime[:] #np.linspace(0, nctime[-1] - nctime[0], nrays)

    # range
    _range = filemetadata('range')
    _range['data'] = ncvars['range'][:]
    _range['meters_to_center_of_first_gate'] = _range['data'][0]
    # assuming the distance between all gates is constant, may not
    # always be true.
    _range['meters_between_gates'] = (_range['data'][1] - _range['data'][0])

    # fields
    # files contain a single corrected reflectivity field
    fields = {}
    
    try:
        ncvars['ZED_HC']
        field_name = filemetadata.get_field_name('ZED_HC')
        field_dic = filemetadata(field_name)
        field_dic['_FillValue'] = ncvars['ZED_HC']._FillValue
        field_dic['units'] = ncvars['ZED_HC'].units
        field_dic['data'] = ncvars['ZED_HC'][:]
        field_dic['applied_calibration_offset'] = ncvars['ZED_HC'].applied_calibration_offset
        fields[field_name] = field_dic
    except KeyError:
        print("ZED_HC does not exist")

    try: 
        ncvars['SNR_HC']
        field_name = filemetadata.get_field_name('SNR_HC')
        field_dic = filemetadata(field_name)
        field_dic['_FillValue'] = ncvars['SNR_HC']._FillValue
        field_dic['units'] = ncvars['SNR_HC'].units
        field_dic['data'] = ncvars['SNR_HC'][:]
        fields[field_name] = field_dic
    except KeyError:
        print("SNR_HC doesn't exist")
        
    try:
        ncvars['LDR_C']
        field_name = filemetadata.get_field_name('LDR_C')
        field_dic = filemetadata(field_name)
        field_dic['_FillValue'] = ncvars['LDR_C']._FillValue
        field_dic['units'] = ncvars['LDR_C'].units
        field_dic['data'] = ncvars['LDR_C'][:]
        fields[field_name] = field_dic
    except KeyError:
        print("LDR_C does not exist")
    
    try:
        ncvars['VEL_HC']
        field_name = filemetadata.get_field_name('VEL_HC')
        field_dic = filemetadata(field_name)
        field_dic['_FillValue'] = ncvars['VEL_HC']._FillValue
        field_dic['units'] = ncvars['VEL_HC'].units
        field_dic['data'] = ncvars['VEL_HC'][:]
        fields[field_name] = field_dic
    except KeyError:
        print("VEL_HC does not exist")
    
    try:
        ncvars['VEL_HCD']
        field_name = filemetadata.get_field_name('VEL_HCD')
        field_dic = filemetadata(field_name)
        field_dic['_FillValue'] = ncvars['VEL_HCD']._FillValue
        field_dic['units'] = ncvars['VEL_HCD'].units
        field_dic['data'] = ncvars['VEL_HCD'][:]
        fields[field_name] = field_dic
    except KeyError:
        print("VEL_HCD does not exist")
    
    try:
        ncvars['SPW_HC']
        field_name = filemetadata.get_field_name('SPW_HC')
        field_dic = filemetadata(field_name)
        field_dic['_FillValue'] = ncvars['SPW_HC']._FillValue
        field_dic['units'] = ncvars['SPW_HC'].units
        field_dic['data'] = ncvars['SPW_HC'][:]
        fields[field_name] = field_dic
    except KeyError:
        print("SPW_HC does not exist")
    
    try:
        ncvars['tx_power_eika_H']
        field_name = filemetadata.get_field_name('tx_power_eika_H')
        field_dic = filemetadata(field_name)
        field_dic['_FillValue'] = ncvars['tx_power_eika_H']._FillValue
        field_dic['units'] = ncvars['tx_power_eika_H'].units
        field_dic['data'] = ncvars['tx_power_eika_H'][:]
        fields[field_name] = field_dic
    except KeyError:
        print("tx_power_eika_H does not exist")
    
    try:
        ncvars['tx_power_after_radome_H']
        field_name = filemetadata.get_field_name('tx_power_after_radome_H')
        field_dic = filemetadata(field_name)
        field_dic['_FillValue'] = ncvars['tx_power_after_radome_H']._FillValue
        field_dic['units'] = ncvars['tx_power_after_radome_H'].units
        field_dic['data'] = ncvars['tx_power_after_radome_H'][:]
        fields[field_name] = field_dic
    except KeyError:
        print("tx_power_after_radome_H does not exist")
    
    try:
        ncvars['SPW_H']
        field_name = filemetadata.get_field_name('SPW_H')
        field_dic = filemetadata(field_name)
        field_dic['_FillValue'] = ncvars['SPW_H']._FillValue
        field_dic['units'] = ncvars['SPW_H'].units
        field_dic['data'] = ncvars['SPW_H'][:]
        fields[field_name] = field_dic
    except KeyError:
        print("SPW_H does not exist")
    
    try:
        ncvars['ZDR']
        field_name = filemetadata.get_field_name('ZDR')
        field_dic = filemetadata(field_name)
        field_dic['_FillValue'] = ncvars['ZDR']._FillValue
        field_dic['units'] = ncvars['ZDR'].units
        #field_dic['applied_calibration_offset'] = ncvars['ZDR'].applied_calibration_offset
        field_dic['data'] = ncvars['ZDR'][:]
        fields[field_name] = field_dic
    except KeyError:
        print("ZDR does not exist")
    
    try:
        ncvars['antenna_diameter']
        field_name = filemetadata.get_field_name('antenna_diameter')
        field_dic = filemetadata(field_name)
        field_dic['data'] = ncvars['antenna_diameter'][:]
        fields[field_name] = field_dic
    except KeyError:
        print("antenna_diameter does not exist")
    
    try:
        ncvars['beamwidthH']
        field_name = filemetadata.get_field_name('beamwidthH')
        field_dic = filemetadata(field_name)
        field_dic['data'] = ncvars['beamwidthH'][:]
        fields[field_name] = field_dic
    except KeyError:
        print("beamwidthH does not exist")
    
    try:
        ncvars['beamwidthV']
        field_name = filemetadata.get_field_name('beamwidthV')
        field_dic = filemetadata(field_name)
        field_dic['data'] = ncvars['beamwidthV'][:]
        fields[field_name] = field_dic
    except KeyError:
        print("beamwidthV does not exist")
    
    try:
        ncvars['clock']
        field_name = filemetadata.get_field_name('clock')
        field_dic = filemetadata(field_name)
        field_dic['data'] = ncvars['clock'][:]
        fields[field_name] = field_dic
    except KeyError:
        print("clock does not exist")
    
    try:
        ncvars['file_state']
        field_name = filemetadata.get_field_name('file_state')
        field_dic = filemetadata(field_name)
        field_dic['data'] = ncvars['file_state'][:]
        fields[field_name] = field_dic
    except KeyError:
        print("file_state does not exist")
    
    try:
        ncvars['frequency']
        field_name = filemetadata.get_field_name('frequency')
        field_dic = filemetadata(field_name)
        field_dic['data'] = ncvars['frequency'][:]
        fields[field_name] = field_dic
    except KeyError:
        print("frequency does not exist")
    
    try:
        ncvars['prf']
        field_name = filemetadata.get_field_name('prf')
        field_dic = filemetadata(field_name)
        field_dic['data'] = ncvars['prf'][:]
        fields[field_name] = field_dic
    except KeyError:
        print("prf does not exist")
    
    try:
        ncvars['pulse_period']
        field_name = filemetadata.get_field_name('pulse_period')
        field_dic = filemetadata(field_name)
        field_dic['data'] = ncvars['pulse_period'][:]
        fields[field_name] = field_dic
    except KeyError:
        print("pulse_period does not exist")
    
    try:
        ncvars['transmit_power']
        field_name = filemetadata.get_field_name('transmit_power')
        field_dic = filemetadata(field_name)
        field_dic['data'] = ncvars['transmit_power'][:]
        fields[field_name] = field_dic
    except KeyError:
        print("transmit_power does not exist")

    # metadata
    metadata = filemetadata('metadata')
    for k in ['institution', 'title', 'used_algorithms']:
        if k in ncobj.ncattrs():
            metadata[k] = ncobj.getncattr(k)

    # latitude, longitude, altitude
    latitude = filemetadata('latitude')
    longitude = filemetadata('longitude')
    altitude = filemetadata('height')
    latitude['data'] = ncvars['latitude'][:] 
    longitude['data'] = ncvars['longitude'][:] 
    altitude['data'] = ncvars['height'][:]

    # sweep parameters
    # sweep_number, sweep_mode, fixed_angle, sweep_start_ray_index,
    # sweep_end_ray_index
    sweep_number = filemetadata('sweep_number')
    sweep_mode = filemetadata('sweep_mode')
    fixed_angle = filemetadata('fixed_angle')
    sweep_start_ray_index = filemetadata('sweep_start_ray_index')
    sweep_end_ray_index = filemetadata('sweep_end_ray_index')

    sweep_number['data'] = np.arange(1, dtype='int32')
    sweep_mode['data'] = np.array(1 * ['fix']) #np.array(1 * ['azimuth_surveillance'])
    #fixed_angle['data'] = np.array(round(ncvars['azimuth'][0],2), dtype='int32')#np.array([0], dtype='float32')
    fixed_angle['data'] = np.array([round(ncvars['azimuth'][0],2)], dtype='float32')#np.array([0], dtype='float32')
    sweep_start_ray_index['data'] = np.array([0], dtype='int32')
    sweep_end_ray_index['data'] = np.array([nrays-1], dtype='int32')

    # azimuth, elevation
    azimuth = filemetadata('azimuth')
    elevation = filemetadata('elevation')

    azimuth['data'] = ncvars['azimuth'][:]
    elevation['data'] = ncvars['elevation'][:]#np.array([0.], dtype='float32')

    metadata['instrument_name']='Copernicus'

    # instrument parameters
    instrument_parameters = None

    return Radar(
        time, _range, fields, metadata, scan_type,
        latitude, longitude, altitude,
        sweep_number, sweep_mode, fixed_angle, sweep_start_ray_index,
        sweep_end_ray_index,
        azimuth, elevation, 
        instrument_parameters=instrument_parameters)

    metadata['instrument_name']='Copernicus'

       # instrument parameters
    instrument_parameters = None

    print(len(azimuth['data']))


    return Radar(
        time, _range, fields, metadata, scan_type,
        latitude, longitude, altitude,
        sweep_number, sweep_mode, fixed_angle, sweep_start_ray_index,
        sweep_end_ray_index,
        azimuth, elevation,time_coverage_start,time_coverage_end,
        instrument_parameters=instrument_parameters)

#------------------------------------------------------------------------
# Define function to produce cfradial files from mmclx files
# ------------------------------------------------------------------------
def copernicus2cfradial(rawfile):
"""Convert Copernicus Chilbolton NetCDF format to CFRadial."""
    user = getpass.getuser()

    print('Opening NetCDF Copernicus raw file ' + ncfile)
    dataset = nc4.Dataset(rawfile,'r+',format='NETCDF4')

    oldhistory = dataset.history;

    dataset.close();

    radar =  read_copernicus(rawfile);

    # -----------------------------------
    # Write cfradial file using arm_pyart
    # -----------------------------------
    cfradfile=ncfile.replace(".nc","-cfrad.nc4");

    pyart.io.cfradial.write_cfradial(cfradfile, radar, format='NETCDF4',
        time_reference=True)

    proctime = datetime.utcnow()
    proctimestr = proctime.ctime()

    history = proctimestr + (" - user:" + user
    + " machine: " + socket.gethostname()
    + " program: copernicus_utils.py create_cfradial_file"
    + " version:" + str(module_version))

    nc1 = nc4.Dataset(cfradfile,'r+',format='NETCDF4')

    if oldhistory.endswith('\n'):
        nc1.history = oldhistory + history;
    else:
        nc1.history = oldhistory + '\n' + history;

    nc1.close();



# ------------------------------------------------------------------------------
# Quicklook generation
# ------------------------------------------------------------------------------
def make_quicklooks(ncfile,figpath):
"""Make quicklooks"""
    user = getpass.getuser()

    Figurename=ncfile.replace(".nc",".png");

    print('Opening NetCDF file ' + ncfile)
    dataset = nc4.Dataset(ncfile,'r+',format='NETCDF3_CLASSIC')

    scantype = dataset.getncattr('scantype');

    dataset.close();

    # --------------------------------
    # Open NetCDF file using arm_pyart
    # --------------------------------
    if (scantype=='RHI'):
        radar = pyart.aux_io.read_camra_rhi(ncfile);
    elif (scantype=='PPI'):
        radar = pyart.aux_io.read_camra_ppi(ncfile);

    dealias_data = pyart.correct.dealias_region_based(radar,
                          ref_vel_field=None, interval_splits=3,
                          interval_limits=None, skip_between_rays=100,
                          skip_along_ray=100, centered=True, nyquist_vel=14.90923,
                          check_nyquist_uniform=True, gatefilter=False,
                          rays_wrap_around=None, keep_original=False, set_limits=True,
                          vel_field='VEL_HV')

    radar.add_field('VEL_UHV', dealias_data)

    from matplotlib import rcParams

    # Define paramters from package
    rcParams['axes.labelsize'] = 16
    rcParams['axes.titlesize'] = 16
    rcParams['xtick.labelsize'] = 14
    rcParams['ytick.labelsize'] = 14

    # create a plot of the first and sixth sweeps
    fig = plt.figure(figsize=(25, 35))
    display = pyart.graph.RadarDisplay(radar)

    ax1 = fig.add_subplot(421)
    display.plot('ZED_H', 0, vmin=-10, vmax=60, ax=ax1,cmap='pyart_HomeyerRainbow',colorbar_orient='horizontal')
    plt.grid()
    ax1.set_ylim(0,12)

    ax2 = fig.add_subplot(423)
    display.plot('ZDR', 0, vmin=-5, vmax=5,ax=ax2,cmap='pyart_HomeyerRainbow',colorbar_orient='horizontal')
    plt.grid()
    ax2.set_ylim(0,12)

    ax3 = fig.add_subplot(425)
    display.plot('LDR', 0, vmin=-35, vmax=5,ax=ax3,cmap='pyart_HomeyerRainbow',colorbar_orient='horizontal')
    plt.grid()
    ax3.set_ylim(0,12)

    ax4 = fig.add_subplot(427)
    display.plot('PDP', 0, vmin=-5, vmax=60, ax=ax4,cmap='pyart_HomeyerRainbow',colorbar_orient='horizontal')
    plt.grid()
    ax4.set_ylim(0,12)

    ax5 = fig.add_subplot(422)
    display.plot('VEL_HV', 0, vmin=-15, vmax=15, ax=ax5,cmap='RdYlBu_r',colorbar_orient='horizontal')
    plt.grid()
    ax5.set_ylim(0,12)

    ax6 = fig.add_subplot(424)
    display.plot('VEL_UHV', 0, vmin=-30, vmax=30, ax=ax6,cmap='RdYlBu_r',colorbar_orient='horizontal')
    plt.grid()
    ax6.set_ylim(0,12)

    ax7 = fig.add_subplot(426)
    display.plot('SPW_HV', 0, vmin=0, vmax=5, ax=ax7,cmap='pyart_HomeyerRainbow',colorbar_orient='horizontal')
    plt.grid()
    ax7.set_ylim(0,12)

    ## Playing with Data and Masks
    CXC=radar.fields['CXC']['data']

    new_CXC=np.ma.masked_where(CXC.data<0.0, CXC)

    L = -np.log10(1-ma.sqrt(new_CXC))

    radar.add_field_like('CXC', 'L', L,replace_existing=True)


    ax8 = fig.add_subplot(428)
    display.plot('L', 0, vmin=0, vmax=3, ax=ax8,cmap='pyart_HomeyerRainbow',colorbar_orient='horizontal')
    plt.grid()
    ax8.set_ylim(0,12)

    plt.savefig(os.path.join(figpath,Figurename),dpi=200)

    plt.close()


# ------------------------------------------------------------------------------
# Quicklook generation
# ------------------------------------------------------------------------------
def make_quicklooks_cfradial(cfradfile,figpath):

    user = getpass.getuser()

    Figurename=cfradfile.replace(".nc",".png");


    # --------------------------------
    # Open NetCDF file using arm_pyart
    # --------------------------------
    radar = pyart.io.read_cfradial(cfradfile);


#   dealias_data = pyart.correct.dealias_region_based(radar,
#                          ref_vel_field=None, interval_splits=3,
#                          interval_limits=None, skip_between_rays=100,
#                          skip_along_ray=100, centered=True, nyquist_vel=14.90923,
#                          check_nyquist_uniform=True, gatefilter=False,
#                          rays_wrap_around=None, keep_original=False, set_limits=True,
#                          vel_field='VEL_HV')

#    radar.add_field('VEL_UHV', dealias_data)

    from matplotlib import rcParams

    # Define paramters from package
    rcParams['axes.labelsize'] = 16
    rcParams['axes.titlesize'] = 16
    rcParams['xtick.labelsize'] = 14
    rcParams['ytick.labelsize'] = 14

    # create a plot of the first and sixth sweeps
    fig = plt.figure(figsize=(25, 35))
    display = pyart.graph.RadarDisplay(radar)

    ax1 = fig.add_subplot(421)
    display.plot('ZED_H', 0, vmin=-10, vmax=60, ax=ax1,cmap='pyart_HomeyerRainbow',colorbar_orient='horizontal')
    plt.grid()
    ax1.set_ylim(0,12)

    ax2 = fig.add_subplot(423)
    display.plot('ZDR', 0, vmin=-5, vmax=5,ax=ax2,cmap='pyart_HomeyerRainbow',colorbar_orient='horizontal')
    plt.grid()
    ax2.set_ylim(0,12)

    ax3 = fig.add_subplot(425)
    display.plot('LDR', 0, vmin=-35, vmax=5,ax=ax3,cmap='pyart_HomeyerRainbow',colorbar_orient='horizontal')
    plt.grid()
    ax3.set_ylim(0,12)

    ax4 = fig.add_subplot(427)
    display.plot('PDP', 0, vmin=-5, vmax=60, ax=ax4,cmap='pyart_HomeyerRainbow',colorbar_orient='horizontal')
    plt.grid()
    ax4.set_ylim(0,12)

    ax5 = fig.add_subplot(422)
    display.plot('VEL_HV', 0, vmin=-15, vmax=15, ax=ax5,cmap='RdYlBu_r',colorbar_orient='horizontal')
    plt.grid()
    ax5.set_ylim(0,12)

    ax6 = fig.add_subplot(424)
    display.plot('VEL_UHV', 0, vmin=-30, vmax=30, ax=ax6,cmap='RdYlBu_r',colorbar_orient='horizontal')
    plt.grid()
    ax6.set_ylim(0,12)

    ax7 = fig.add_subplot(426)
    display.plot('SPW_HV', 0, vmin=0, vmax=5, ax=ax7,cmap='pyart_HomeyerRainbow',colorbar_orient='horizontal')
    plt.grid()
    ax7.set_ylim(0,12)

    ## Playing with Data and Masks
    CXC=radar.fields['CXC']['data']

    new_CXC=np.ma.masked_where(CXC.data<0.0, CXC)

    L = -np.log10(1-ma.sqrt(new_CXC))

    radar.add_field_like('CXC', 'L', L,replace_existing=True)


    ax8 = fig.add_subplot(428)
    display.plot('L', 0, vmin=0, vmax=3, ax=ax8,cmap='pyart_HomeyerRainbow',colorbar_orient='horizontal')
    plt.grid()
    ax8.set_ylim(0,12)

    plt.savefig(os.path.join(figpath,Figurename),dpi=200)

    plt.close()


if __name__ == "__main__":
    import sys, PythonCall
    PythonCall.PythonCall(sys.argv).execute()
