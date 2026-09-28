# Analises complementares do relatorio: IC de Wilson, McNemar, CV 10x10 com
# t de Nadeau-Bengio, sensibilidade a semente/particao, curva de perda e varredura de k.
import json, time, numpy as np, pandas as pd, warnings
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import load_iris, load_wine
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
RS=42
def knn(): return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
def mlp(seed=RS): return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(10,10),activation="relu",solver="adam",max_iter=1000,random_state=seed))
out={}
splits={}
bases={"Iris":load_iris(),"Wine":load_wine()}
fig,axes=plt.subplots(1,2,figsize=(10,3.6))
for (nb,b),ax in zip(bases.items(),axes):
    X,y=b.data,b.target
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.3,stratify=y,random_state=RS)
    r={"n":len(y),"d":X.shape[1],"classes":list(map(str,b.target_names)),"dist":np.bincount(y).tolist(),"ntr":len(ytr),"nte":len(yte)}
    preds={}
    for nm,f in [("KNN",knn),("MLP",mlp)]:
        m=f().fit(Xtr,ytr); p=m.predict(Xte); preds[nm]=p
        pr,rc,f1,s=precision_recall_fscore_support(yte,p)
        acc=accuracy_score(yte,p); k=int((p==yte).sum()); n=len(yte)
        lo,hi=stats.binomtest(k,n).proportion_ci(method="wilson")
        d={"acc":acc,"prec_cls":pr.tolist(),"rec_cls":rc.tolist(),"f1_cls":f1.tolist(),"sup":s.tolist(),
           "f1_macro":f1.mean(),"cm":confusion_matrix(yte,p).tolist(),"wilson":[lo,hi],"errors":n-k}
        if nm=="MLP":
            c=m[-1]; d["n_iter"]=int(c.n_iter_); d["loss"]=float(c.loss_)
            d["params"]=int(sum(w.size for w in c.coefs_)+sum(bb.size for bb in c.intercepts_))
            ax.plot(c.loss_curve_,label=nb); 
        r[nm]=d
    # McNemar exato
    a=(preds["KNN"]==yte); bm=(preds["MLP"]==yte)
    b01=int((a&~bm).sum()); b10=int((~a&bm).sum())
    r["mcnemar"]={"knn_only":b01,"mlp_only":b10,"p":stats.binomtest(b01,b01+b10,0.5).pvalue if b01+b10>0 else 1.0}
    # CV repetida 10x10 pareada + t corrigido Nadeau-Bengio
    cv=RepeatedStratifiedKFold(n_splits=10,n_repeats=10,random_state=RS)
    sk=cross_val_score(knn(),X,y,cv=cv); sm=cross_val_score(mlp(),X,y,cv=cv)
    diff=sm-sk; J=len(diff); ratio=1/9
    tcor=diff.mean()/np.sqrt((1/J+ratio)*diff.var(ddof=1)) if diff.var()>0 else 0
    pcor=2*stats.t.sf(abs(tcor),J-1)
    r["rcv"]={"knn_mean":sk.mean(),"knn_std":sk.std(),"mlp_mean":sm.mean(),"mlp_std":sm.std(),"diff_mean":diff.mean(),"t":tcor,"p":pcor,
              "wins_mlp":int((diff>0).sum()),"ties":int((diff==0).sum()),"wins_knn":int((diff<0).sum())}
    # sensibilidade à semente (split fixo)
    accs=[accuracy_score(yte,mlp(s).fit(Xtr,ytr).predict(Xte)) for s in range(30)]
    r["seed"]={"mean":float(np.mean(accs)),"std":float(np.std(accs)),"min":float(min(accs)),"max":float(max(accs))}
    # sensibilidade do split (30 splits)
    ks,ms=[],[]
    for s in range(30):
        a1,a2,b1,b2=train_test_split(X,y,test_size=.3,stratify=y,random_state=s)
        ks.append(accuracy_score(b2,knn().fit(a1,b1).predict(a2))); ms.append(accuracy_score(b2,mlp().fit(a1,b1).predict(a2)))
    splits[nb]=(ks,ms)
    r["split"]={"knn_mean":np.mean(ks),"knn_std":np.std(ks),"mlp_mean":np.mean(ms),"mlp_std":np.std(ms),"acc42_rank_knn":float(stats.percentileofscore(ks,r["KNN"]["acc"])),}
    # latência de inferência
    for nm,f in [("KNN",knn),("MLP",mlp)]:
        m=f().fit(Xtr,ytr); t=time.perf_counter()
        for _ in range(200): m.predict(Xte)
        r[nm]["pred_ms"]=(time.perf_counter()-t)/200*1000
    # k sweep
    r["ksweep"]={k:float(cross_val_score(make_pipeline(StandardScaler(),KNeighborsClassifier(k)),X,y,cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=RS)).mean()) for k in [1,3,5,7,9,11,15]}
    out[nb]=r
    ax.set_title(f"Curva de perda (log-loss) — MLP — {nb}"); ax.set_xlabel("Época"); ax.set_ylabel("Perda de treino"); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("figuras/curva_perda_mlp.png",dpi=160)
# robustez a 30 particoes treino/teste
fig,axes=plt.subplots(1,2,figsize=(10,3.6),sharey=True)
for ax,(nb,(ks,ms)) in zip(axes,splits.items()):
    xs=np.arange(len(ks))
    ax.plot(xs,ks,"o-",ms=3,lw=1,label="KNN (k=5)")
    ax.plot(xs,ms,"s-",ms=3,lw=1,label="MLP (10,10)")
    ax.set_title(f"Acurácia em 30 partições — {nb}"); ax.set_xlabel("random_state do split"); ax.grid(alpha=.3)
axes[0].set_ylabel("Acurácia (teste)"); axes[0].legend()
plt.tight_layout(); plt.savefig("figuras/robustez_splits.png",dpi=160)
json.dump(out,open("resultados_complementares.json","w"),indent=1,default=float)
print("Resultados salvos em resultados_complementares.json")
