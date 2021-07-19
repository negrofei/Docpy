__all__ = ['WRF',
           'Dato',
           'ERAI',
           'ERA5',
           ]

########################################## CLASE DE DATOS ###########################################
class Dato:
    
    def __init__(self, ruta):
        """
        Class of precipitation dataset. At the moment only CMORPH-8km y MSWEP-0.1°
        
        - ruta: str, path to the directory of the netcdf file
        
        """
        import os
        from netCDF4 import Dataset
        import numpy as np
        datos = ['cmorph','mswep','estaciones', 'imerg']
        self.ruta = os.path.abspath(os.path.split(ruta)[0])
        self.filename = os.path.split(ruta)[1]
        self.data_netcdf = Dataset(os.path.join(self.ruta,self.filename))
        # diccionarios de los datos a utilizar
        self.name = datos[[d in os.path.join(self.ruta,self.filename).lower() for d in datos].index(True)]
        
        if self.name == 'cmorph':
            self.precip_name = 'cmorph'
            if '8km' in self.filename:
                self.dominio = '8km'
            elif '0.25deg' in self.filename:
                self.dominio = '0.25deg'
        elif self.name == 'mswep':
            self.precip_name = 'precipitation'
            self.dominio = '0.1deg'
            
        elif self.name == 'estaciones':
            self.precip_name = 'precip'
            self.dominio = '0.25deg'

        elif self.name == 'imerg':
            self.precip_name = 'precipitationCal'
            self.dominio = '0.1deg'
        
        
    def timedata(self):
        """ 
        - return[0]: list, list of times of the simulation in format YYYY-mm-dd HH:MM:SS
        - return[1]: int, delta of time in hours 
        
        """
        from netCDF4 import num2date
        tiempo = self.data_netcdf.variables['time']
        # --- cmorph, estaciones y imerg
        if (self.name == 'cmorph') or (self.name == 'estaciones') or (self.name == 'imerg'):
            time_list = [str(num2date(tiempo[i], units=tiempo.units, calendar=tiempo.calendar)) for i in range(len(tiempo[:]))]
        # --- mswep
        elif self.name == 'mswep':
            time_list = [str(num2date(tiempo[i], units=tiempo.units)) for i in range(len(tiempo[:]))]
        delta_t = num2date(tiempo[1], units=tiempo.units)-num2date(tiempo[0], units=tiempo.units)
        delta_t_hs = delta_t.days*24+delta_t.seconds/3600    
        # --- imerg
        
        return time_list, delta_t_hs

    def get_latlon(self):
        """
        - return: numpy.array, array of latitudes and longitudes
        """
        lon = self.data_netcdf.variables['lon'][:]
        if lon[0]>=0:
            lon = lon-360
            
        lat = self.data_netcdf.variables['lat'][:]
        if lat[0]>lat[1]:
            lat = lat[::-1]
            
        return lat, lon
    
    def precip_selection(self):
        """
        - return: numpy.array, array of precipitation dataset with dims (time, lat, lon)
        """
        from netCDF4 import Dataset
        import os
        lat = self.data_netcdf.variables['lat'][:]
        if lat[0]>lat[1]:
            s = slice(None, None, -1)
        else:
            s = slice(None, None, 1)
        if self.name == 'cmorph':
            precip = self.data_netcdf.variables[self.precip_name][:,0,s,:]
        elif self.name == 'mswep':
            precip = self.data_netcdf.variables[self.precip_name][:,s,:]
        elif self.name == 'estaciones':
            precip = self.data_netcdf.variables[self.precip_name][:,s,:]
        elif self.name == 'imerg':
            precip = self.data_netcdf.variables[self.precip_name][:,s,:]
        return precip

########################################## CLASE DE WRF ############################################

class WRF:
    def __init__(self, ruta):
        """ 
        CIMA class for WRF simualtions 
        
        - ruta: str, path to wrf simulations
        
        """
        import os
        from netCDF4 import Dataset
        import numpy as np
        self.ruta = os.path.abspath(os.path.split(ruta)[0])
        self.filename = os.path.split(ruta)[1]
        self.name = 'wrf'
        self.data_netcdf = Dataset(os.path.join(self.ruta,self.filename))
        if np.diff(self.data_netcdf.variables['lon'][:]).min()>0.1:
            self.dominio = 'd01'
        else:
            self.dominio = 'd02'
            
    def timedata(self):
        """ Devuelve una lista con los tiempos de la simulacion """
        from netCDF4 import num2date
        tiempo = self.data_netcdf.variables['time']
        time_list = [str(num2date(tiempo[i], units=tiempo.units, calendar=tiempo.calendar)) for i in range(len(tiempo[:]))]
        delta_t_hs = num2date(tiempo[1]-tiempo[0], units=tiempo.units).hour
        return time_list, delta_t_hs
    
    def get_latlon(self, box=None):
        lon = self.data_netcdf.variables['lon'][:]
        lat = self.data_netcdf.variables['lat'][:]
        if box:
            import numpy as np
            # setting box 
            assert len(box) == 4, "box should be a 4 size list"
            lonmin, lonmax, latmin, latmax = box
            x_idx_ini = int(np.where(np.abs(lon-lonmin)==np.min(np.abs(lon-lonmin)))[0])
            x_idx_fin = int(np.where(np.abs(lon-lonmax)==np.min(np.abs(lon-lonmax)))[0])+1
            y_idx_ini = int(np.where(np.abs(lat-latmin)==np.min(np.abs(lat-latmin)))[0])
            y_idx_fin = int(np.where(np.abs(lat-latmax)==np.min(np.abs(lat-latmax)))[0])+1
            
            if y_idx_ini>y_idx_fin:
                y_idx_ini, y_idx_fin = y_idx_fin, y_idx_ini
                
            lon = lon[x_idx_ini:x_idx_fin]
            lat = lat[y_idx_ini:y_idx_fin]


        return lat, lon
    
    def get_levels(self):
        """
        """
        lev = self.data_netcdf.variables['lev'][:]
        return lev
    
    def get_variables(self, var, box=None):
        import numpy as np
        """ Funcion que obtiene las variables
        Voy a robar una clase del wrf-python para poder asignarle más de un nombre a una variable
        """
        class either(object):
            def __init__(self, *varnames):
                self.varnames = varnames
            def __call__(self, data_netcdf):
                for varname in self.varnames:
                    if varname in data_netcdf.variables:
                        return varname
                raise ValueError("{} are not valid variable names".format(
                    self.varnames))
        zonales = ('umet','ua')
        meridio = ('vmet','va')
        if (var in zonales):
            var = either('umet','ua')(self.data_netcdf)
        elif (var in meridio):
            var = either('vmet','va')(self.data_netcdf)
        vari = self.data_netcdf.variables[var][:]
        if (var in zonales) or (var in meridio):
            vari = np.ma.masked_where(np.abs(vari)>1e+20, vari)

        if box:
            # setting box 
            assert len(box) == 4, "box should be a 4 size list"
            lonmin, lonmax, latmin, latmax = box
            lat, lon = self.get_latlon()
                        
            x_idx_ini = int(np.where(np.abs(lon-lonmin)==np.min(np.abs(lon-lonmin)))[0])
            x_idx_fin = int(np.where(np.abs(lon-lonmax)==np.min(np.abs(lon-lonmax)))[0])+1
            y_idx_ini = int(np.where(np.abs(lat-latmin)==np.min(np.abs(lat-latmin)))[0])
            y_idx_fin = int(np.where(np.abs(lat-latmax)==np.min(np.abs(lat-latmax)))[0])+1
            
            if y_idx_ini>y_idx_fin:
                y_idx_ini, y_idx_fin = y_idx_fin, y_idx_ini
                
            vari = vari[...,x_idx_ini:x_idx_fin]
            vari = vari[...,y_idx_ini:y_idx_fin,:]

            
        return vari


    def acum_time(self,acum):
        time_list, delta_t_hs = self.timedata()
        if acum < delta_t_hs:
            import sys
            sys.exit(' '.join(['#### ERROR: acumulation',str(acum),'hs is smaller than dt',str(delta_t_hs),'####']))
        inter = int(acum/delta_t_hs)                                                                  
        return time_list[::inter]

########################################## CLASE DE ERAI ############################################



class ERAI:
    def __init__(self, ruta):
        """
        - ruta: str, path to ERAI netCDF file

        """
        import os
        from netCDF4 import Dataset, num2date
        import numpy as np
        self.ruta = ruta
        self.dvars_era = {'CAPE':'var59',
                          'CFHI':'var84',
                          'HUS':'var133',
                          'U':'var131',
                          'V':'var132',
                          'Z':'var129'}
        self.name = 'erai'
        self.filename = os.path.split(ruta)[-1].split('.nc')[0]
        self.var = self.filename.split('_')[0]
        self.caso = self.filename.split('_')[-1]
        self.data_netcdf = Dataset(os.path.join(self.ruta))
        
    def timedata(self):
        """ 
        - return[0]: list, list of times of the simulation in format YYYY-mm-dd HH:MM:SS
        - return[1]: int, delta of time in hours 
        
        """
        from netCDF4 import num2date
        tiempo = self.data_netcdf.variables['time']
        
        time_list = [str(num2date(tiempo[i], units=tiempo.units, calendar=tiempo.calendar)) for i in range(len(tiempo[:]))]
        delta_t = num2date(tiempo[1], units=tiempo.units)-num2date(tiempo[0], units=tiempo.units)
        delta_t_hs = delta_t.days*24+delta_t.seconds/3600    
        return time_list, delta_t_hs


    def get_latlon(self):
        """
        - return: numpy.array, array of latitudes and longitudes
        """
        lon = self.data_netcdf.variables['lon'][:]
        if lon[0]>=0:
            lon = lon-360
            
        lat = self.data_netcdf.variables['lat'][:]
        if lat[0]>lat[1]:
            lat = lat[::-1]
        return lat, lon    
            
    def get_levels(self):
        """
        - return: numpy.array, array of vertical levels

        """
        if self.name == 'erai':
            lev = self.data_netcdf.variables['plev'][:]/100
        else:
            lev = self.data_netcdf.variables['plev'][:]
        return lev

    def get_variables(self,var):
        """ 
        - return: numpy.array, array with the selected field

        """
        var = self.data_netcdf.variables[self.dvars_era[var]][:,:,::-1,:]
        return var

       
class ERA5:
    def __init__(self, ruta):
        """
        - ruta: str, path to ERA5 netCDF file

        """
        import os
        from netCDF4 import Dataset, num2date
        import numpy as np
        self.ruta = ruta
        self.name = 'era5'
        self.filename = os.path.split(ruta)[-1].split('.nc')[0]
        self.data_netcdf = Dataset(os.path.join(self.ruta))
        
    def timedata(self):
        """ 
        - return[0]: list, list of times of the simulation in format YYYY-mm-dd HH:MM:SS
        - return[1]: int, delta of time in hours 
        
        """
        from netCDF4 import num2date
        tiempo = self.data_netcdf.variables['time']
        
        time_list = [str(num2date(tiempo[i], units=tiempo.units, calendar=tiempo.calendar)) for i in range(len(tiempo[:]))]
        delta_t = num2date(tiempo[1], units=tiempo.units)-num2date(tiempo[0], units=tiempo.units)
        delta_t_hs = delta_t.days*24+delta_t.seconds/3600    
        return time_list, delta_t_hs


    def get_latlon(self, box=None):
        """
        - return: numpy.array, array of latitudes and longitudes
        """
        lon = self.data_netcdf.variables['longitude'][:]
        if lon[0]>=0:
            lon = lon-360
            
        lat = self.data_netcdf.variables['latitude'][:]
        if lat[0]>lat[1]:
            lat = lat[::-1]
        
        if box:
            import numpy as np
            # setting box 
            assert len(box) == 4, "box should be a 4 size list"
            lonmin, lonmax, latmin, latmax = box
            x_idx_ini = int(np.where(np.abs(lon-lonmin)==np.min(np.abs(lon-lonmin)))[0])
            x_idx_fin = int(np.where(np.abs(lon-lonmax)==np.min(np.abs(lon-lonmax)))[0])+1
            y_idx_ini = int(np.where(np.abs(lat-latmin)==np.min(np.abs(lat-latmin)))[0])
            y_idx_fin = int(np.where(np.abs(lat-latmax)==np.min(np.abs(lat-latmax)))[0])+1
            
            if y_idx_ini>y_idx_fin:
                y_idx_ini, y_idx_fin = y_idx_fin, y_idx_ini
                
            lon = lon[x_idx_ini:x_idx_fin]
            lat = lat[y_idx_ini:y_idx_fin]
        return lat, lon    
            
    def get_levels(self):
        """
        - return: numpy.array, array of vertical levels

        """
        lev = self.data_netcdf.variables['level'][::-1]
        return lev

    def get_variables(self,var, box=None):
        """ 
        - return: numpy.array, array with the selected field

        """
        keys = [*self.data_netcdf.variables.keys()]
        if var in keys[4:]:
            pass
        else:
            while var not in keys:
                var = input('Select variable from the following:\n'+'\t'.join([k for k in keys[4:]])+'\n')
        vari = self.data_netcdf.variables[var][:,::-1,::-1,:]
        
        if box:
            # setting box
            import numpy as np
            assert len(box) == 4, "box should be a 4 size list"
            lonmin, lonmax, latmin, latmax = box
            lat, lon = self.get_latlon()
                        
            x_idx_ini = int(np.where(np.abs(lon-lonmin)==np.min(np.abs(lon-lonmin)))[0])
            x_idx_fin = int(np.where(np.abs(lon-lonmax)==np.min(np.abs(lon-lonmax)))[0])+1
            y_idx_ini = int(np.where(np.abs(lat-latmin)==np.min(np.abs(lat-latmin)))[0])
            y_idx_fin = int(np.where(np.abs(lat-latmax)==np.min(np.abs(lat-latmax)))[0])+1
            
            if y_idx_ini>y_idx_fin:
                y_idx_ini, y_idx_fin = y_idx_fin, y_idx_ini
                
            vari = vari[...,x_idx_ini:x_idx_fin]
            vari = vari[...,y_idx_ini:y_idx_fin,:]
        return vari




