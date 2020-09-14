
import  numpy as np
def calc_cnr(E,solmat,oneE):
    #N=len(E)
    if oneE==1:
        cnr=np.zeros(solmat.shape[1])
        rms=np.zeros(solmat.shape[1])
        for j in range(solmat.shape[1]):
            sol=solmat[:,j]
            thresh=0.8*np.max(sol)
            idxe=np.where(sol>thresh)
            
            idxb=np.where(sol<thresh)        
            EE=sol[idxe[0]]
            BB=sol[idxb[0]]
            #print('index of E:',np.mean(EE))
            print('index of EB:',np.mean(EE)-np.mean(BB))
            cnr[j]=10*np.log10(2*(np.mean(EE)-np.mean(BB))**2/(np.var(EE)+np.var(BB)))
            rms[j]=np.sqrt(np.mean(np.abs(2*(sol-E)/(E+sol))**2))#np.sqrt(np.mean(np.abs((sol-Etrue)/Etrue)**2))
    else:
        N=E.shape[1]
        cnr=np.zeros(N)
        rms=np.zeros(N)
        for j in range(N):
            sol=solmat[:,j]
            Ei=E[:,j]
            thresh=np.max(Ei)*0.41/0.5
            idxe=np.where(sol>thresh)
            idxb=np.where(sol<thresh)        
            EE=sol[idxe[0]]
            BB=sol[idxb[0]]
            cnr[j]=10*np.log10(2*(np.mean(EE)-np.mean(BB))**2/(np.var(EE)+np.var(BB)))
            rms[j]=np.sqrt(np.mean(np.abs(2*(sol-Ei)/(sol+Ei))**2))#np.sqrt(np.mean(np.abs((sol-Etrue)/Etrue)**2))
    return cnr,rms