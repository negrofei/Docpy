
__all__ = ['common_doms',
           'common_times',
           'utils',
           'acum_time',
           'printer',
           'quick_map',
           'delta_time_runs',
           'nodes_ppn_runs',
           'get_namelist_info',
           'latlon_idx',
           'get_poweroften',
           'OOMFormatter',
          ]

def printer(string, num_char=100,char='#'):
    print(('{:'+char+'^'+str(num_char)+'}').format(' '.join([' ',string,' '])))
    return None

def common_doms(paths):
    """
    paths: list, list of strings of file paths of netcdf files to calculate common domains within simulations and/or datasets 

    return: box = [lonmin, lonmax, latmin, latmax]
    """
    import Docpy
    import os
    datos = ['cmorph', 'mswep']

    lats = []
    lons = []
    for path in paths:
        print(path)
        if any(d in path.lower() for d in datos) == True:
            case = Docpy.Dato(path)
        elif 'wrf' in path:
            case = Docpy.WRF(path)

        lat, lon = case.get_latlon()

        lats.append(lat)
        lons.append(lon)

    lonmin = max([min(lons[i]) for i in range(len(lons))])
    lonmax = min([max(lons[i]) for i in range(len(lons))])
    latmin = max([min(lats[i]) for i in range(len(lats))])
    latmax = min([max(lats[i]) for i in range(len(lats))])

    print('Dominio comun:','\n',                        
          'lonmin',lonmin,'\n',                        
          'lonmax',lonmax,'\n',                        
          'latmin',latmin,'\n',                        
          'latmax',latmax,'\n')            
    
    box = [lonmin, lonmax, latmin, latmax]
    return box


def common_times(lista_de_tiempos):        
    """             
    Funcion que determina los elementos comunes entre las listas dentro de la lista                 
    """
    if len(lista_de_tiempos)>1:

        lista = lista_de_tiempos
        try:
            common = list(set(lista[0]).intersection(lista[1]))                        
        except:
            common = list(set(lista[0][0]).intersection(lista[1]))
        for l in range(1,len(lista)):                                    
            common = list(set(common).intersection(lista[l]))                                        
        common.sort()                                           
        return common
    elif len(lista_de_tiempos)==1:
        return lista_de_tiempos

def acum_time(case,acum):
    """
    - case: WRF or Dato class object 
    - acum: int, step in hours to shorten the list of times
    **
    - return: list, list of times by acum steps. 
    """
    time_list, delta_t_hs = case.timedata()                        
    if acum < delta_t_hs:                                        
        import sys                                                    
        sys.exit(' '.join(['#### ERROR: acumulation',str(acum),'hs is smaller than dt',str(delta_t_hs),'####']))
    inter = int(acum/delta_t_hs)                                                                    
    return time_list[::inter]

def quick_map(lon,lat, field):
    import os
    import cartopy.crs as ccrs
    import cartopy.io.shapereader as shpreader
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER 
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import numpy as np
    #provincias = shpreader.Reader('/home/martin.feijoo/provincias/provincias.shp')
    bordes = cfeature.NaturalEarthFeature(          # limites de los paises
            category='cultural',
            name='admin_0_boundary_lines_land',
            scale='50m',
            edgecolor='k',
            facecolor='none'
            )

    fig = plt.figure(figsize=(10,6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    cs = ax.contourf(lon, lat, field, transform=ccrs.PlateCarree())
    ax.add_feature( bordes )
    #for rec in provincias.records():
    #    ax.add_geometries( [rec.geometry], ccrs.PlateCarree(), edgecolor="k", facecolor='none', linewidth=0.5)
    cbar = plt.colorbar(cs)
    gl = ax.gridlines(draw_labels=True,color='black',alpha=0.2,linestyle='--')
    gl.xlabels_top =False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}
    ax.coastlines(resolution='50m')
    ax.set_extent([min(lon), max(lon), min(lat), max(lat)])
    plt.show()
    return None

def delta_time_runs(path='./run_wrf.log'):
    from datetime import datetime
    with open(path,'r') as f:
        lines = f.readlines()
    t_ini = datetime.strptime(lines[0][:-1], '%a %b %d %H:%M:%S ART %Y')
    t_fin = datetime.strptime(lines[-1][:-1], '%a %b %d %H:%M:%S ART %Y')
    delta = t_fin-t_ini
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print('{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds)))
    return hours, minutes, seconds

def nodes_ppn_runs(path='./run_wrf.pbs'):
    with open(path, 'r') as f:
        lines = f.readlines()
    string = [line for line in lines if 'nodes' in line][0]
    nodes = int(string.split(':')[0][-1])
    p = string.split(':')[1]
    ppn = int(p[-3:-1])
    return nodes, ppn

def get_namelist_info(path='./namelist.input'):
    with open(path,'r') as f:
        lines = f.read().splitlines()
    doms_line = [line for line in lines if 'max_dom' in line][0]
    nro_doms = int(doms_line.split(',')[0].split('=')[1])
    resolution = []
    Npoints = []
    timestep = []
    history = []
    for N in range(nro_doms):
        res_line = [line for line in lines if 'dx' in line][0].split('=')[1]
        resolution.append( int(res_line.split(',')[N]) )
        pointx_line = [line for line in lines if 'e_we' in line][0].split('=')[1]
        pointy_line = [line for line in lines if 'e_sn' in line][0].split('=')[1]
        Npoints.append( int(pointx_line.split(',')[N])*int(pointy_line.split(',')[N]) )
        time_line = [line for line in lines if 'time_step' in line][0].split('=')[1]
        ratio_line = [line for line in lines if 'parent_grid_ratio' in line][0].split('=')[1]
        timestep.append( int(int(time_line.split(',')[0])/int(ratio_line.split(',')[N])) )
        hist_line = [line for line in lines if 'history_interval' in line][0].split('=')[1]
        history.append( int(hist_line.split(',')[0]) ) 
    return [resolution, Npoints, timestep, history]    

#def save_fig(fig_id, IMAGES_PATH=ruta_figs, tight_layout=True, fig_extension="png", resolution=300):
#    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
#    print("Saving figure", fig_id)
#    if tight_layout:
#        plt.tight_layout()
#    plt.savefig(path, format=fig_extension, dpi=resolution)

def latlon_idx(lat, lon, box):
    """
    Funcion que busca los indices correspondientes a la box para los arrays de lat y lon

    - lat: numpy.array 
    - lon:  numpy.array
    - box: list, [lonmin, lonmax, latmin, latmax]

    - return: [imin, imax, jmin, jmax]
    """
    import numpy as np
    imin = int(np.where(np.abs(lon-box[0]) == np.min(np.abs(lon-box[0])))[0])
    imax = int(np.where(np.abs(lon-box[1]) == np.min(np.abs(lon-box[1])))[0])
    jmin = int(np.where(np.abs(lat-box[2]) == np.min(np.abs(lat-box[2])))[0])
    jmax = int(np.where(np.abs(lat-box[3]) == np.min(np.abs(lat-box[3])))[0])
    return imin, imax, jmin, jmax


import matplotlib.ticker as mtick
class OOMFormatter(mtick.ScalarFormatter): #por algun motivo que no entiendo esto no anda
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        mtick.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self, nothing):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin, vmax):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % mtick._mathdefault(self.format)

def get_poweroften(m):
    """
    Funcion que te determina el orden del numero entre 0 y 1.
    """
    if m==0:
        print('m es 0')
    elif int(m)>0:
        print('m es mayor a 1')
        i=0
        if int(m/100)>0:
            while int(m/10)>0:
                m = m/10
                i += 1
    else:
        i = 0
        while int(m)==0:
            m = m*10
            i -= 1
    return i

utils = {'box': [-65.9654, -52.6832, -36.535, -25.279],
         #'box': [-65.9654, -52.6832, -37.0476, -25.279],
         'box15': [-65.9654, -50.1452, -34.9963, -22.9669],
         }

