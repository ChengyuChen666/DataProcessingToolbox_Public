'''
Lorentz line shape data analysis toolbox.

version = 1.0.0
'''
__version__='v1.0.0'
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks,find_peaks_cwt,savgol_filter
from scipy.optimize import curve_fit
'''
===============
Basic Functions
===============
'''
def Normalize_10(A):
    '''
    Normalize the data to an interval of [0,1].
    
    Parameters
    ----------
    A : array_like
        Input array (data).
    
    Returns
    -------
    B : narray
        Normalized data with maximun of 1 and minimun of 0. 
    '''
    B=(A-np.min(A))/(np.max(A)-np.min(A))
    return B

def Normalize_pm1(A):
    '''
    Normalize the data to an interval of [-1,1].
    
    Parameters
    ----------
    A : array_like
        Input array (data).
    
    Returns
    -------
    B : narray
        Normalized data with maximun of 1 and minimun of -1. 
    '''
    B=(2*A-np.min(A)-np.max(A))/(np.max(A)-np.min(A))
    return B

def DataFilting(Data,
                win_width=10,
                MaxOrder=3,
                IfShow=False):
    '''
    Filter the data by using savgol filter.
    
    Parameters
    ----------
    Data : array_like
        Input array (data).
    
    win_width : int, optional
        Filting window length.
    
    MaxOrder : int, optional
        Order of savgol filter.
    
    IfShow : bool, optional
        Whether the data manipulation will be visually displayed.

    Returns
    -------
    Data_fl : narray
        Output.
    
    '''
    Data_fl=savgol_filter(Data,win_width,MaxOrder)
    if IfShow:
        plt.plot(Data,'--')
        plt.plot(Data_fl)
        plt.legend(['Raw data','Filtered data'])
        plt.show()
    return Data_fl

def DC_Filtering(A):
    '''
    Normalize and filter the data by removing the direct current term.
    
    Parameters
    ----------
    A : array_like
        Input array (data).
    
    Returns
    -------
    B : narray
       output data.
    '''
    B=Normalize_pm1(A)
    Bw=np.fft.fft(B)
    Bw[0]=0.
    Bw[-1]=0.
    b=np.fft.ifft(Bw)
    return b

def DataAlignment(a,v,
              peaks_th=1,
              IfShow=True):
    '''
    Align two data arrays by shifting and cutting.
    
    Parameters
    ----------
    a, v : array_like
        Input array (data).

    peaks_th : int, optional
        Align the two data according to the peaks_th maximun of the corralation.
    
    IfShow : bool, optional
        Whether the data manipulation will be visually displayed.
    
    Returns
    -------
    new_a, new_v : narray
       output data.
    '''
    #Get correlation
    A=DC_Filtering(a)
    V=DC_Filtering(v)
    N=len(a)
    cov=np.correlate(A,V,'full')
    vPosition=np.arange(-len(a)+1,len(v))

    #finding and reordering peaks
    peaks, _ = find_peaks(cov,distance=100,height=np.max(cov)*0.5)
    cov_peaks=cov[peaks]
    vPosition_peaks=vPosition[peaks]
    maxindex=np.argsort(cov_peaks)
    vPosition_peaks=vPosition_peaks[maxindex]
    cov_peaks=cov_peaks[maxindex]

    #shift data according to peaks
    k=vPosition_peaks[-peaks_th]
    print('Correlation maximun: k = '+str(k))
    if k>=0:
        new_a=a[k:]
        new_v=v[:-k]
    else:
        new_a=a[:k]
        new_v=v[-k:]
     
    #input Peaks number
    if IfShow:
        fig,ax=plt.subplots(3,1)
        ax[0].plot(a)
        ax[0].plot(v)
        ax[0].legend(['a','v'])

        ax[1].plot(vPosition,cov)
        ax[1].plot(vPosition_peaks,cov_peaks,'r.')
        for i in range(0,len(cov_peaks)):
            ax[1].text(vPosition_peaks[-i-1],cov_peaks[-i-1],str(i+1))

        NN=len(new_a[:N_cut])
        lam=np.linspace(0,6e-4*NN,NN)+1520
        ax[2].plot(lam,new_a)
        ax[2].plot(lam,new_v)
        ax[2].legend(['new_a','new_v'])
        plt.show()


    #output
    M=len(new_a)
    new_a=np.hstack((new_a,np.zeros(N-M)))
    new_v=np.hstack((new_v,np.zeros(N-M)))
    return new_a,new_v

'''
============
Peak Finding
============
'''
def Lorentz(points,a):
    '''
    Lorentz line shape function. L=1/(1+x^2)
    
    Parameters
    ----------
    points : int
        Number of points.
    
    a: float
        Width of the lorentz line shape. 
    
    Returns
    -------
    L : narray
       output.
    '''
    vec = np.arange(0, points) - (points - 1.0) / 2
    x=vec/(a/2)
    L=1/(1+x**2)/np.pi
    return L

def CWT_FindPeak(Data,
                 n_width_min=5,
                 n_width_max=50,
                 n_width_Grid=100,
                 IfShow=False):
    '''
    Find the locations of Lorentz line shape in the data by using the method of continuous wavelet transformation.
    
    Parameters
    ----------
    Data : array_like
        Input array (data).

    n_width_min : int, optional
        The minimun width of Lorentz line shape.
    
    n_width_max : int, optional
        The maxmun width of Lorentz line shape.
    
    n_width_Grid: int, optional
        Width of the lorentz line shape. 

    IfShow : bool, optional
        Whether the data manipulation will be visually displayed.
    
    Returns
    -------
    peakin : narray
       The index of the locations of Lorentz line shape in the data.
    '''
    B=-Normalize_10(Data)+1
    A=DataFilting(B,IfShow=False)
    n=np.arange(0,len(A))
    widths=np.linspace(n_width_min,n_width_max,n_width_Grid)
    peakind=find_peaks_cwt(A,widths,Lorentz)
    if IfShow:
        Transmission=Normalize_10(Data)
        Transmission_fl=-A+1
        plt.plot(Transmission,'*-')
        plt.plot(Transmission_fl,'--')
        plt.plot(n[peakind],Transmission[peakind],'or')
        plt.legend(['Raw data','Filtered data','Peaks'])
        plt.show()
    return peakind

'''
=========
Q Fitting
=========
'''
def Standard_Trans(lam,width,lam0,Tmin,Tmax):
    '''
    Give a standard transimission of Lorentz line shape.
    f(lambda)=Tmax*(1-(1-Tmin/Tmax)/(1+x^2)), x=(lambda-lam0)/(width/2)

    Arg:
        lam: wavelength (unit: nm)
        width: Full-Width-Half-Maximum (unit: nm)
        lam0: center wavelength (unit: nm)
        Tmin: minmun transimission
        Tmax: maximun transimission
    '''
    x=(lam-lam0)/(width/2)
    L=Tmax*(1-(1-Tmin/Tmax)/(1+x**2))
    return L

class Mode:
    'Class of optical mode'

    def __init__(self,lam_center,data_transimission,data_lam):
        '''
        Creat a mode object.
        
        Arg:
            lam_center: location of the mode (unit: nm)
            data_transimission: transimission data of this mode (unit: a.u.)
            data_lam: lam data of this mode (unit: nm)
        '''
        self.lam_center=lam_center
        self.data_transimission=data_transimission
        self.data_lam=data_lam
        self.IfFitted=False
        self.FWHM=[]
        self.Q=[]
        self.Qerr=[]
        self.lam_center_fit=[]
        self.Tmin=[]
        self.Tmax=[]
        self.Fitting_transimission=np.array([])

    def Display(self):
        '''
        Display the raw data and fitting function of the mode.
        '''
        fig,ax=plt.subplots()
        ax.plot(self.data_lam,self.data_transimission,'*-')
        if self.IfFitted:
            ax.plot(self.data_lam,self.Fitting_transimission)
            title=('Q = '+str(round(self.Q))+' | FWHM = '+str(np.round(self.FWHM*1e3,3))+' pm')
            ax.set_title(title)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Transimission (a.u.)')
        ax.set_xlim([np.min(self.data_lam),np.max(self.data_lam)])
        ax.set_ylim([0,1])
        plt.show()

    def Fitting(self,
                GuessQ=1e5,
                IfReport=False):
        '''
        Fit the Lorentz lines shape by least squares methods.
        
        Parameters
        ----------
        GuessQ: float
            Initial guess for quality factor Q.

        IfReport : bool, optional
            Whether report the fitting result.
        
        Returns
        -------
        IfSucceed: bool
            If fitting succeds, return true.
            
        '''
        try:
            popt,pcov=curve_fit(Standard_Trans,
                                self.data_lam,self.data_transimission,
                                p0=(self.lam_center/GuessQ,self.lam_center,0,1))
            self.FWHM=np.abs(popt[0])
            self.Q=self.lam_center/self.FWHM
            self.Qerr=np.abs(self.lam_center/self.FWHM**2*pcov[0,0])
            self.lam_center_fit=popt[1]
            self.Tmin=popt[2]
            self.Tmax=popt[3]
            self.Fitting_transimission=Standard_Trans(self.data_lam,popt[0],popt[1],popt[2],popt[3])
            self.IfFitted=True
            if IfReport:
                print('-------------------------')
                print('Lorentz Fitting Report')
                print('lambda_0 = '+str(round(self.lam_center_fit,3))+'nm')
                print('Q = '+str(round(self.Q)))
                print('Q_err = '+str(round(self.Qerr)))
                print('FWHM = '+str(round(self.FWHM*1e3,3))+'pm')
                print('Transimission_min = '+str(round(self.Tmin,3)))
                print('Transimission_max = '+str(round(self.Tmax,3)))
                print('-------------------------')
            return True
        except:
            if IfReport:
                print('Fiting failed')
            return False

'''
===================================
Data Segmentation and Batch Fitting
===================================
'''
#This part is not very good. It will be impoved in the future.
def Data_Segmentation(peak_ind,Data_lam,Data_transimission):
    '''
    Segmentate data according to peak index peak_ind

    Parameters
    ----------
    peak_ind : int
        Locations of Lorentz line shape (Peaks index) in the data.

    Data_lam : array_like
        Data of wavelength. (unit: nm)
    
    Data_tranismission : array_like
        Data of transmission spectrum. (unit: a.u.)

    Returns
    -------
    Mode_list : narray
        The mode objects list.
    '''
    M=len(peak_ind)
    Mode_list=[]
    
    #Segmentation
    Begin_Index=0
    End_Index=int((peak_ind[0]+peak_ind[1])/2)
    Opt_mode=Mode(Data_lam[peak_ind[0]],
                  Data_transimission[Begin_Index:End_Index],
                  Data_lam[Begin_Index:End_Index])
    Mode_list.append(Opt_mode)

    for i in range(1,M-1):
        Begin_Index=int((peak_ind[i]+peak_ind[i-1])/2)
        End_Index=int((peak_ind[i]+peak_ind[i+1])/2)
        Opt_mode=Mode(Data_lam[peak_ind[i]],
                      Data_transimission[Begin_Index:End_Index],
                      Data_lam[Begin_Index:End_Index])
        Mode_list.append(Opt_mode)

    Begin_Index=int((peak_ind[M-2]+peak_ind[M-1])/2)
    Opt_mode=Mode(Data_lam[peak_ind[M-1]],
                  Data_transimission[Begin_Index:],
                  Data_lam[Begin_Index:])
    Mode_list.append(Opt_mode)

    return Mode_list

def Data_Batch_Fitting(Mode_list,
                 IfFiltrate=True,
                 Qmax=5e6,
                 Qmin=1e4,
                 IfShow=False):
    '''
    Batch fitting of an mode list.

    Parameters
    ----------
    Mode_list : array_like
        The mode objects list.

    IfFiltrate : bool, optional
        Determint wheter filter the data.
    
    Qmax : float
        The maximun of Q.

    Qmin : 
        The minimun of Q.

    IfShow : bool, optional
        Whether the data manipulation will be visually displayed.
    
    Returns
    -------
    mode_Q_list: narray
        Q values of the input mode list.

    mode_lambda_list: narray
        Wavelength values of the input mode list. (unit: nm)

    mode_number_list: narray
        Mode's numbers values of the input mode list. (unit: nm)

    Max_Q_index: int
        The index of mode which has max Q value. 
    '''
    M=len(Mode_list)
    mode_Q_list=np.array([])
    mode_lambda_list=np.array([])
    mode_number_list=np.array([],dtype=int)
    print('----------------')
    print('Batch fitting starts.')
    for i in range(0,M):
        IfSuceceed=Mode_list[i].Fitting()
        if IfSuceceed:
            mode_lambda_list=np.append(mode_lambda_list,Mode_list[i].lam_center)
            mode_Q_list=np.append(mode_Q_list,Mode_list[i].Q)
            mode_number_list=np.append(mode_number_list,i)
        else:
            print('Mode No. '+str(i)+' fitting fails.')
    print('Batch fitting ended.')
    print('----------------')
    #Filtration of Q
    if IfFiltrate:
        Correct_Index= mode_Q_list >= Qmin#lower bound
        mode_Q_list=mode_Q_list[Correct_Index]
        mode_lambda_list=mode_lambda_list[Correct_Index]
        mode_number_list= mode_number_list[Correct_Index]
        Correct_Index= mode_Q_list <= Qmax#upper bound
        mode_Q_list=mode_Q_list[Correct_Index]
        mode_lambda_list=mode_lambda_list[Correct_Index]
        mode_number_list= mode_number_list[Correct_Index]
    
    #Find maximun of Q
    Max_Q_index=np.argmax(mode_Q_list)
    print('----------------')
    print('Maximun Q is: '+str(round(mode_Q_list[Max_Q_index])))
    print('Mode index of maximun Q is: '+str(mode_number_list[Max_Q_index]))
    print('----------------')

    #Plotting
    if IfShow:
        fig,ax=plt.subplots()
        ax.stem(mode_lambda_list,mode_Q_list,'*')
        ax.set_xlabel('wavelentgh (nm)')
        ax.set_ylabel('Quality factor Q')
        # ax.set_yscale('log')
        plt.show()
    return mode_Q_list,mode_lambda_list,mode_number_list,Max_Q_index