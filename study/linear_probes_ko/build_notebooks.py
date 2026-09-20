"""Rebuild the Korean teaching notebooks from reviewable text; does not run models."""
from pathlib import Path
import textwrap
import nbformat as nbf

HERE = Path(__file__).resolve().parent


def md(s):
    return nbf.v4.new_markdown_cell(textwrap.dedent(s).strip())


def code(s):
    return nbf.v4.new_code_cell(textwrap.dedent(s).strip())


def save(name, cells):
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "ARENA · Vast PyTorch", "language": "python", "name": "arena-vast"},
        "language_info": {"name": "python", "version": "3.12"},
    })
    nbf.write(nb, HERE / name)


SETUP = '''
from pathlib import Path
import sys
HERE = Path.cwd()
if not (HERE / "probe_lab.py").exists():
    candidates = [HERE / "study/linear_probes_ko", HERE / "ARENA_3.0/study/linear_probes_ko"]
    HERE = next(p for p in candidates if (p / "probe_lab.py").exists())
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from probe_lab import (group_split_indices, fit_mean_difference, fit_logistic_probe,
                       score_metrics, fit_train_pca, group_bootstrap_ci)
np.set_printoptions(precision=3, suppress=True)
'''


def main():
    save("00_START_HERE.ipynb", [
        md('''
        # Linear probes: 잘 맞는 분류기에서 믿을 수 있는 실험으로

        **대상:** ARENA 1.2를 마쳤고 residual stream·hook·activation patching을 아는 학습자.
        **목표:** probe를 구현하는 것보다, 결과가 무엇을 입증하고 무엇을 입증하지 않는지 판단하기.
        원본 `[1.3.1] Linear Probes`를 한국어 개인 학습용으로 재구성했다. 원본은 보존했다.

        시작 질문: **held-out accuracy 95%인 truth probe를 발견했다. 이 모델을 거짓말 탐지기로 써도 되는가?**
        지금 답을 두 문장으로 써 두고 마지막에 고쳐 보자.
        '''),
        code('''
        first_answer = ""  # 관측 사실 / 추가로 필요한 증거를 각각 한 문장으로
        '''),
        md('''
        ## 진행 순서 — 약 4~6시간, 중간에 멈춰도 됨

        | 순서 | 파일 | 핵심 실험 질문 | 예상 시간 |
        |---|---|---|---|
        | 1 | `01_design_and_probes.ipynb` | 높은 점수가 label 대신 shortcut을 읽은 결과라면? | 60~90분 |
        | 2 | `02_real_truth_experiment.ipynb` | 사실·주제가 바뀌어도 같은 방향이 통하는가? | 60~90분 |
        | 3 | `03_intervention_and_claims.ipynb` | 읽을 수 있다는 것과 출력에 영향을 준다는 것은 같은가? | 45~60분 |
        | 4 | `04_deception_attention.ipynb` | truth, 지시문, 실제 deception, high-stakes를 어떻게 구분하는가? | 60~90분 |

        각 노트북은 **예측 → 실험 → 반례 → 허용되는 결론** 순서다. 출력 전에 예측을 적어야 실험이 학습이 된다.
        막히면 해당 질문의 접힌 힌트를 먼저 보고, `probe_lab.py`는 구현 참고용으로 연다.
        기본 학습은 저장된 실제 activation과 작은 합성 실험으로 진행한다. 13B 재실행은 선택이다.
        '''),
        md('''
        ## 먼저 합의할 언어

        - **사실의 참/거짓:** 외부 정답에 비추어 라벨을 정한다. 모델의 믿음 자체는 아니다.
        - **선형적으로 읽힘:** 지정한 데이터·위치·층에서 선형 분류기가 미지 표본을 구분한다.
        - **일반화:** 새 entity, 주제, template, 모델 중 무엇이 바뀌었는지를 명시해야 한다.
        - **개입 민감성:** 해당 activation 조작에 출력이 달라진다. 자연 실행의 유일한 인과 메커니즘이라는 뜻은 아니다.
        - **기만:** false utterance와 동의어가 아니다. 진실만 골라 말해도 오도할 수 있다.

        연구의 최신성은 `SOURCES_AND_CLAIMS.md`에 정리했다. **원본 코드 버그**, **논문의 적용 범위**, **후속 반례**는 서로 구분한다.
        한 후속 논문의 성공/실패를 학계 전체의 합의나 완전한 debunk로 부르지 않는다.
        '''),
        code(SETUP),
        code('''
        import torch, json, platform
        print("Python:", platform.python_version(), "Torch:", torch.__version__)
        print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU: 캐시 분석은 가능")
        print("실제 실험 결과:", (HERE / "artifacts/results.json").exists())
        print("실습 폴더:", HERE)
        '''),
        md('''
        ## Vast에서 열기 / 중단 / 복귀

        VS Code의 커널 선택에서 **ARENA · Vast PyTorch** 또는 `/venv/main/bin/python`을 고른다.
        파일을 저장하면 VS Code 종료 후에도 서버 디스크에는 남는다. 서버의 recycle/destroy는 디스크를 지운다.
        GPU 모델은 노트북마다 동시에 여러 개 올리지 않는다. 캐시 분석에는 모델이 필요 없다.

        현재 챕터는 OpenAI API를 호출하지 않는다. Llama 추론은 이 Vast GPU에서 한다.
        HF 토큰은 코드/노트북/`.env`에 넣지 않았다. 이미 다운로드한 모델은 `local_files_only=True`로 읽는다.
        다른 인스턴스에서 다시 다운로드할 때만 HF 로그인이 필요하다.

        **중간 점검:** 다음 실험을 하기 전 “split 단위 / 선택에 쓰는 데이터 / 단 하나의 주장 / 그 주장을 깨는 결과”를 기록한다.
        '''),
    ])

    save("01_design_and_probes.ipynb", [
        md('''
        # 1. 프로브를 만들기 전에 실험을 설계하기

        **오늘의 주장 후보:** “모델 내부에 진실 방향이 있다.”
        이것을 검증 가능한 좁은 문장으로 바꿔 보자:
        “고정한 모델·층·token 위치의 activation으로 학습한 선형 분류기가, 학습에서 보지 않은 도시의 문장 참/거짓을 구분한다.”

        `x ∈ R^d → score = wᵀx+b`. 지금 필요한 수학은 이 정도다. 중요한 것은 x, y, split을 어떻게 만들었는가다.
        '''),
        code(SETUP),
        md('''
        ## 실험 1 — 원본 행 단위 split이 충분한가?

        “Paris is in France” / “Paris is in Brazil” / 두 문장의 부정형은 서로 독립된 사실이 아니다.
        **도시 단위로 분할하고, 그 도시의 모든 파생 문장은 같은 분할에 둔다.**
        pair split은 문장 암기 경로 하나를 줄인다. 모든 confound를 제거하는 것은 아니다.

        **예측:** 같은 도시가 train/test 양쪽에 있을 때 어떤 정보로 정답을 맞출 수 있을까?
        '''),
        code('''
        groups = np.repeat(np.arange(100), 2)
        y = np.tile([0, 1], 100)
        split = group_split_indices(groups, seed=42)
        for a,b in [(split.train,split.val),(split.train,split.test),(split.val,split.test)]:
            assert set(groups[a]).isdisjoint(groups[b])
        display(pd.DataFrame([{"split":k,"rows":len(getattr(split,k)),
                                "groups":len(np.unique(groups[getattr(split,k)]))}
                              for k in ["train","val","test"]]))
        '''),
        md('''
        ## 실험 2 — validation에서 잘 맞아도 OOD에서 실패하는 이유

        합성 activation의 첫 축에는 약한 label 신호, 둘째 축에는 강한 shortcut을 심는다.
        train/validation에서는 shortcut이 정답과 함께 움직이고 OOD에서는 반대로 움직인다.
        **LLM에서 관찰한 결과가 아니라, 실험의 식별 한계를 직접 보이는 반례다.**

        먼저 예측: validation으로 C를 잘 골랐다면 이 OOD 실패도 막을 수 있을까?
        '''),
        code('''
        rng = np.random.default_rng(42)
        def make_data(n, shortcut_sign):
            y = np.tile([0,1], n//2)
            signal = 2*y-1
            X = rng.normal(size=(n, 12))
            X[:,0] += .8*signal
            X[:,1] = 2.5*shortcut_sign*signal + rng.normal(scale=.3,size=n)
            return X,y
        Xtr,ytr=make_data(400,1)
        Xva,yva=make_data(200,1)
        Xod,yod=make_data(200,-1)
        rows=[]
        for C in [.01,.1,1]:
            p=fit_logistic_probe(Xtr,ytr,C=C)
            rows.append({"C":C,"validation_AUROC":score_metrics(yva,p.decision_function(Xva))["auroc"],
                         "OOD_AUROC":score_metrics(yod,p.decision_function(Xod))["auroc"]})
        display(pd.DataFrame(rows))
        '''),
        md('''
        <details><summary>해석 힌트</summary>
        Validation은 선택 편향을 막지만, validation에 없는 분포 변화까지 보장하지 않는다.
        OOD AUROC가 0.5 아래여도 test를 본 뒤 score 부호를 뒤집어 성공이라고 보고하면 안 된다.
        train의 positive label 의미를 고정한 채 실패 자체를 보고한다.
        </details>

        ## 실험 3 — difference of means의 빠뜨리기 쉬운 절편

        `w = μ₁ − μ₀`, `b = −wᵀ(μ₁+μ₀)/2`.
        이 결정 경계는 두 평균의 중점을 지난다. `x @ w > 0`만 쓰면 activation 원점에 의존한다.
        **연습:** x와 두 평균을 같은 벡터만큼 평행 이동할 때 판정이 왜 같아야 하는지 식으로 설명하자.
        '''),
        code('''
        p = fit_mean_difference(Xtr,ytr)
        shift = np.full(Xtr.shape[1],17.)
        shifted = fit_mean_difference(Xtr+shift,ytr)
        assert np.allclose(p.decision_function(Xva),shifted.decision_function(Xva+shift),atol=1e-7)
        print("중점 보정 후 평행 이동에 불변:", True)
        print("raw score는 확률이 아님. sigmoid를 취해도 자동으로 calibration되지는 않음.")
        '''),
        md('''
        ## 실험 4 — 표준화된 LR 가중치를 residual stream에 더해도 되는가?

        학습 좌표 `z=(x−μ)/σ`의 `w_scaled`와 원래 activation의 방향은 다르다.
        `w_raw = w_scaled/σ`, `b_raw = b_scaled − w_rawᵀμ`.
        예측 함수·cosine 비교·개입이 모두 같은 좌표를 쓰는지 검증한다.

        **예측:** AUROC가 그대로인데 cosine 또는 steering 결과가 바뀔 수 있는 이유는?
        '''),
        code('''
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        scaler=StandardScaler().fit(Xtr)
        lr=LogisticRegression(C=.1,max_iter=2000,random_state=0).fit(scaler.transform(Xtr),ytr)
        wraw=lr.coef_[0]/scaler.scale_
        braw=lr.intercept_[0]-wraw@scaler.mean_
        assert np.allclose(Xva@wraw+braw,lr.decision_function(scaler.transform(Xva)))
        print("좌표 변환 최대 오차:",np.max(np.abs(Xva@wraw+braw-lr.decision_function(scaler.transform(Xva)))))
        '''),
        md('''
        ## 실험 5 — PCA 그림을 어느 정도 믿을까?

        PCA는 큰 분산을 찾는다. class separation 목적함수가 아니다.
        train에만 fit하고 held-out 점을 투영한다. 라벨 색칠은 지도 정보다.
        PCA 2D에 안 보인다고 고차원 선형 분리가 없다는 결론도 성립하지 않는다.
        '''),
        code('''
        pca=fit_train_pca(Xtr,2)
        fig,axs=plt.subplots(1,2,figsize=(10,3))
        for ax,X,y,title in [(axs[0],Xva,yva,"validation"),(axs[1],Xod,yod,"OOD: shortcut reversed")]:
            Z=pca.transform(X)
            ax.scatter(Z[:,0],Z[:,1],c=y,cmap="coolwarm",s=10,alpha=.6)
            ax.set(title=title,xlabel="PC1",ylabel="PC2")
        plt.tight_layout(); plt.show()
        '''),
        md('''
        ## 보조 반례 — 선형 probe 실패가 정보 부재를 뜻하지 않는 이유

        XOR: 두 bit를 그대로 보유하지만, 둘의 XOR label은 단일 직선으로 나눌 수 없다.
        비선형 probe가 성공하면 모델의 정보인지 probe가 새 계산을 했는지를 추가로 따져야 한다.
        CCS의 consistency objective도 정답 의미를 유일하게 식별해 주지는 않는다.
        지금은 CCS 구현보다 이 **식별 가능성 문제**를 이해하면 충분하다. 자세한 논쟁은 출처 문서의 선택 읽기.
        '''),
        code('''
        Xxor=np.tile(np.array([[-1,-1],[-1,1],[1,-1],[1,1]]),(40,1))
        yxor=(Xxor[:,0]*Xxor[:,1]<0).astype(int)
        print("선형 LR:",score_metrics(yxor,fit_logistic_probe(Xxor,yxor).decision_function(Xxor)))
        print("두 bit로 정답을 복원:", np.mean((Xxor[:,0]*Xxor[:,1]<0)==yxor))
        '''),
        md('''
        ## 다음 실험 사전 등록 — 아래를 채운 뒤 02로

        1. 최종 test를 열기 전에 선택할 것: 층, LR의 C, threshold, token 위치 중 무엇인가?
        2. 도시 이름 split과 다른 주제 OOD는 각각 어떤 주장을 검사하는가?
        3. token 길이 baseline이 강하면 어떤 대조군을 추가할 것인가?
        4. 높은 IID AUROC / 낮은 OOD AUROC를 한 문장으로 정직하게 보고하라.
        '''),
        code('''
        preregistration = {
            "claim": "",
            "unit_of_split": "",
            "selection_data": "",
            "negative_control": "",
            "result_that_would_change_my_mind": "",
        }
        '''),
    ])

    save("02_real_truth_experiment.ipynb", [
        md('''
        # 2. 실제 Llama-2-13B의 truth representations

        원본 Geometry of Truth CSV와 **Llama-2-13B BF16의 실제 block 출력**을 사용한다.
        실험 조건은 `artifacts/results.json`, 각 행과 split은 `data_manifest.csv`에 저장되어 있다.
        논문 전체 재현이 아니라 작은 단일 seed pilot이다. “예상 정확도”를 정답으로 두지 않는다.

        **예측 먼저:** cities → neg_cities, 번역, 수 비교 중 어디서 전이가 가장 약할까? 이유는?
        '''),
        code(SETUP),
        code('''
        import json
        ART=HERE/"artifacts"
        assert (ART/"results.json").exists(), "README의 실제 실험 재실행 안내를 먼저 보세요."
        result=json.loads((ART/"results.json").read_text())
        df=pd.read_csv(ART/"data_manifest.csv")
        acts=np.load(ART/"activations.npz",allow_pickle=False)
        display(df.groupby(["domain","split"]).agg(rows=("label","size"),groups=("group","nunique"),true_fraction=("label","mean")))
        print({k:result[k] for k in ["model","model_revision","seed","layers","gpu","dtype","max_gpu_GiB"]})
        '''),
        md('''
        ## 무엇을 고정했는가?

        - **학습:** cities의 120개 도시 중 60%; validation 20%, test 20%. 동일 도시의 참/거짓 쌍은 같이 간다.
        - **선택:** `[8,14,20,28,36]`번 block과 LR C `[.01,.1,1]`만 validation AUROC로 선택한다. MM도 같은 층 후보를 쓴다.
        - **위치:** 마침표를 포함한 문장의 마지막 유효 token. right padding, truncation 없이 확인.
        - **OOD:** neg_cities는 test 도시만; 번역은 스페인어 단어 단위, 수 비교는 unordered number pair 단위로 표본 선택.
        - **평가:** 선택 완료 후 고정한 분류기를 test/OOD에 적용한다. threshold는 train에서 정해진 logit 0.
        - **대조군:** token 길이 1차원 LR, 고정 layer14에서 20회 train label permutation.

        층 14는 역사적 Llama-2-13B 설정이지 보편적인 truth layer가 아니다.
        HF `hidden_states[-1]`은 마지막 layernorm을 거칠 수 있으므로 **decoder block에 직접 hook**을 걸었다.
        원본처럼 모든 층의 전체 sequence activation을 동시에 보관하지 않는다.
        '''),
        code('''
        train=np.flatnonzero(df.split=="train")
        val=np.flatnonzero(df.split=="val")
        test=np.flatnonzero(df.split=="test")
        assert set(df.iloc[train].group).isdisjoint(df.iloc[val].group)
        assert set(df.iloc[train].group).isdisjoint(df.iloc[test].group)
        selection=pd.read_csv(ART/"validation_selection.csv")
        display(selection)
        print("validation으로 잠근 선택:",result["selected"])
        '''),
        md('''
        **생각할 지점:** MM과 LR을 각자의 best layer에서 비교하면 무엇이 함께 달라지는가?

        <details><summary>힌트</summary>
        classifier와 layer가 동시에 달라진다. 최종 pipeline 비교로는 가능하지만 classifier 자체의 우열을
        주장하려면 같은 layer에서 비교해야 한다. 새로운 비교를 test를 본 뒤 고안했다면 exploratory라고 표시한다.
        </details>
        '''),
        code('''
        final=pd.DataFrame(result["results"])
        display(final[["probe","layer","domain","n","groups","auroc","balanced_accuracy","auroc_ci"]])
        fig,ax=plt.subplots(figsize=(9,4))
        domains=list(df.domain.unique())
        for j,kind in enumerate(["MM","LR"]):
            part=final[final.probe==kind].set_index("domain").loc[domains]
            mean=part.auroc.to_numpy()
            ci=np.array(part.auroc_ci.tolist())
            x=np.arange(len(domains))+(j-.5)*.18
            ax.errorbar(x,mean,yerr=np.maximum(0,np.vstack([mean-ci[:,0],ci[:,1]-mean])),fmt="o",capsize=4,label=kind)
        ax.axhline(.5,color="gray",ls="--")
        ax.set(xticks=np.arange(len(domains)),xticklabels=domains,ylim=(-.03,1.03),ylabel="AUROC (group bootstrap 95% CI)")
        ax.legend();plt.tight_layout();plt.show()
        '''),
        md('''
        ## 신뢰구간은 어떤 불확실성을 포함하는가?

        도시/단어/수 쌍을 단위로 resampling했다. 도시·수 비교의 paired 문장을 독립 표본으로 세지 않았다.
        이번 번역 표본은 60개 단어에 각 한 문장이다.
        이 interval은 **고정된 모델·학습 결과에 조건부인 평가 표본 변동**이다.
        다른 seed, 다른 모델, 재학습, layer 선택의 불확실성까지 포함하지 않는다.
        neg_cities와 cities는 같은 test 도시를 공유하므로 두 평가의 독립성도 가정하지 않는다.
        작은 test에서 완전 분리되면 percentile bootstrap이 [1,1]로 퇴화할 수 있다. 모집단 성능이 확실히 1이라는 뜻은 아니다.

        **연습:** AUROC와 balanced accuracy가 서로 다른 이야기를 한다면 threshold 이동과 순위 능력을 구분해 설명하라.
        '''),
        code('''
        display(pd.DataFrame(result["token_length_baseline"]))
        null=np.array(result["shuffled_label_auroc"])
        plt.figure(figsize=(6,2.5));plt.hist(null,bins=8);plt.axvline(.5,color="black",ls="--")
        plt.xlabel("Held-out AUROC: 20 shuffled-label MM fits at fixed block 14");plt.show()
        print("Null range:",null.min(),null.max())
        '''),
        md('''
        **주의:** null이 낮다고 모든 shortcut이 사라진 것은 아니다. 길이 baseline도 단 하나의 nuisance control이다.
        여기의 label permutation은 행 단위로 쌍 구조를 깨뜨리는 sanity control이며 정식 permutation 유의성 검정이 아니다.
        도시/country 분포, tokenization, template, negation, 모델의 fact familiarity가 남아 있다.

        ## 그림과 정량 결과를 함께 보기
        train-only PCA를 test에 투영한다. 가장 예쁜 PCA 그림을 고른 뒤 이를 독립 증거라고 부르지 않는다.
        '''),
        code('''
        X=acts["layer_14"]
        pca=fit_train_pca(X[train],2)
        Z=pca.transform(X[test]); labels=df.label.to_numpy()
        fig,ax=plt.subplots(figsize=(6,4))
        ax.scatter(Z[:,0],Z[:,1],c=labels[test],cmap="coolwarm",s=30)
        ax.set(title="Unseen cities, block 14; PCA fit on training cities",xlabel="PC1",ylabel="PC2")
        plt.show()
        '''),
        md('''
        ## 직접 해 볼 실험 — 실행 전 설계만 먼저

        A. 같은 entity 분할을 유지한 채 문장을 paraphrase하고 성능을 비교한다.
        B. 마지막 마침표 대신 첫 token을 읽는 대조군을 만든다.
        C. 여러 seed의 train/val split으로 층 선택 안정성을 평가한다.

        **최소 보고서:** 데이터 단위, 선택 절차, 대조군, 결과, 가장 그럴듯한 대안 설명, 다음 검증 하나.
        새 실험은 이 test를 반복 소비하므로 이미 확인한 결과와 분리해 exploratory로 기록하고,
        최종 주장은 별도로 봉인한 새 entity 집합에서 검증한다.

        재실행 코드는 `run_truth_pilot.py`다. 모델 다운로드/학습을 자동으로 시작하는 notebook 셀은 없다.
        '''),
        md('''
        <details><summary>실측 결과 해설 — 먼저 자기 해석을 쓴 뒤 열기</summary>

        이 인스턴스의 seed42 pilot에서 LR는 validation으로 block8/C=.01이 선택되었다.
        city test AUROC≈.998, neg_cities AUROC=0.000이었다. label 의미를 고정하면
        부정문에서 **순위가 뒤집힌 실패**다. test를 본 뒤 부호를 뒤집어 일반화 성공으로 고치면 안 된다.
        지리적 연관성 등을 읽었을 가능성은 다음 실험의 가설이며, 이 결과만으로 메커니즘이 확정되지는 않는다.

        MM block14는 neg_cities AUROC≈.899이나 balanced accuracy=.50이었다.
        순위 정보가 있어도 학습 분포에서 고정한 threshold는 OOD에 맞지 않을 수 있다.
        별도 OOD validation에서 threshold를 보정하는 실험은 가능하지만 **zero-shot transfer와 다른 설정**이다.
        이 작은 결과를 모든 LR/MM 모델의 우열로 일반화하지 않는다.

        </details>
        '''),
        code('''
        report = {"supported_claim":"", "unsupported_claim":"", "alternative_explanation":"", "next_experiment":""}
        '''),
    ])

    save("03_intervention_and_claims.ipynb", [
        md('''
        # 3. 읽을 수 있는 정보와 실제 쓰이는 정보

        **출발 질문:** probe가 100% 맞히면 그 방향으로 activation을 바꿀 때 모델 출력도 바뀌어야 할까?
        “정보가 있다 / 특정 readout이 가능하다 / 모델이 자연 실행에서 그것을 쓴다”를 분리한다.
        '''),
        code(SETUP),
        md('''
        ## 반례를 직접 만들기

        모델 상태에 같은 정보를 두 번 저장하되 실제 출력은 두 번째 좌표만 읽게 만든다.
        첫 좌표의 probe는 완벽하지만 그 좌표 개입은 아무 효과가 없다.
        이 toy example은 Llama에 대한 관찰이 아니라 **논리적 함의가 성립하지 않음**을 보여 준다.
        '''),
        code('''
        rng=np.random.default_rng(8)
        y=np.tile([0,1],100); signal=2*y-1
        H=np.c_[signal,signal+.2*rng.normal(size=len(y))]
        probe_score=H[:,0]
        model_output=lambda h: h[:,1]
        changed=H.copy();changed[:,0]+=5
        print("Probe accuracy:",np.mean((probe_score>0)==y))
        print("첫 좌표 개입 출력 변화:",np.max(np.abs(model_output(changed)-model_output(H))))
        changed[:,1]+=5
        print("둘째 좌표 개입 출력 변화:",np.mean(model_output(changed)-model_output(H)))
        '''),
        md('''
        ## 세 종류의 개입을 혼동하지 않기

        | 실험 | 실제 하는 일 | 바로 말할 수 있는 것 |
        |---|---|---|
        | Additive steering | h ← h + αd | 이 조작에 대한 출력 민감성 |
        | Activation patching | 특정 site를 다른 실행의 activation으로 교체 | 지정된 source/target/site에서의 개입 효과 |
        | 인과 매개 분석 | 명시한 causal model과 contrast에 따른 효과 추정 | 가정이 만족될 때의 매개 효과 |

        원본의 `mm_nie/lr_nie`는 단순 score 변화다. 이 자료에서는 **additive intervention effect**라 부른다.
        patching조차 off-distribution 상태나 사용되지 않던 경로를 만들 수 있다. 따라서 성공한 개입 하나가
        자연 실행의 메커니즘을 유일하게 확정하지 않는다. 핵심 후속 반례는 `SOURCES_AND_CLAIMS.md` 참고.
        '''),
        md('''
        ## 실제 모델 pilot: 사전에 고정한 설계

        - train cities의 block14에서 MM direction을 학습하고 unit norm으로 만든다.
        - 같은 방향의 train projection 표준편차를 scale로 삼아 α ∈ {−2,−1,0,1,2}를 미리 정한다.
        - held-out 6개 도시의 참/거짓 12문장을 고정한 few-shot True/False prompt에 넣는다.
        - 마지막 `Answer:` token의 block14 출력에만 더한다. metric은 **True − False next-token logit**이다.
        - unit norm random direction 하나와 α=0을 대조한다. hook은 항상 제거한다.
        - 답변 token은 고립된 문자열 대신 전체 prompt+답변을 tokenize하여 한 token continuation인지 검증한다.

        **이 설계의 약점도 미리 기록:** probe 학습 위치는 문장 끝, 개입 위치는 few-shot의 Answer:다.
        효과 없음은 truth representation 부재와 동치가 아니다. 효과 있음도 바로 belief 변경은 아니다.
        random direction 하나와 12문장은 탐색적 sanity check 수준이다.
        '''),
        code('''
        import json
        r=json.loads((HERE/"artifacts/results.json").read_text())
        effects=pd.DataFrame(r["intervention"])
        display(effects)
        fig,ax=plt.subplots(figsize=(7,4))
        for name,part in effects.groupby("direction"):
            ax.plot(part.alpha,part.mean_logit_shift,"o-",label=name)
        ax.axhline(0,color="gray",ls="--")
        ax.set(xlabel="alpha × train projection SD",ylabel="Mean change in True − False logit",title="Fixed block 14, 12 held-out statements")
        ax.legend();plt.show()
        '''),
        md('''
        ## 결과에 따라 결론을 다르게 쓰기

        - MM 효과가 random보다 크면: 이 특정 prompt/site/scale에서 방향 선택이 출력 변화에 관련되어 있다.
        - 둘 다 비슷하면: generic perturbation 가능성이 있어 특이성 근거가 약하다.
        - 둘 다 작으면: 위치·맥락 불일치, model readout, BF16 작은 변화 등 대안 설명을 남긴다.
        - 큰 α에서만 효과가 생기면: 자연 activation 범위를 벗어나는지 norm과 출력 품질을 확인한다.

        **지금 하지 않은 검증:** 여러 random 방향 분포, 여러 prompt, source-consistent patching,
        truth와 무관한 출력 손상 지표, per-label 효과, 독립 test replication.
        이 모든 것을 당장 구현할 필요는 없다. 자신의 핵심 주장에 필요한 최소 다음 대조군 하나를 선택한다.
        '''),
        code('''
        conclusion = {
            "observation":"",
            "allowed_claim":"",
            "overclaim_to_avoid":"",
            "most_informative_next_control":"",
        }
        '''),
        md('''
        <details><summary>구두 시험 — 설명할 수 있으면 다음으로</summary>

        1. LR가 MM보다 분류를 잘하지만 steering은 약할 수 있는가? 그렇다. 예측 목적함수와 인과적 사용은 다르며 norm/좌표를 맞춰야 한다.
        2. `w_scaled`를 raw residual에 더하면? 다른 방향을 조작하게 된다.
        3. 음의 α에서 출력이 false로 이동하면 모델이 거짓말한 것인가? 아니다. 특정 출력 점수의 조작 결과이며 기만 의도 라벨이 없다.
        4. α=0 비교가 다른 결과라면? dropout, hook 잔류, padding/site, nondeterminism부터 점검한다.

        </details>

        이제 04에서 truth label을 **instruction label·행동 label·high-stakes label**과 분리한다.
        '''),
    ])


if __name__ == "__main__":
    main()
