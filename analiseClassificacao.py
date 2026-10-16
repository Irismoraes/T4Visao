import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from matplotlib.colors import LinearSegmentedColormap

cores = {"setosa": "#7B2CBF", "versicolor": "#FF69B4", "virginica": "#D62828"}
mapa_cores = LinearSegmentedColormap.from_list("roxo_rosa_vermelho", ["#FFFFFF", "#FF69B4", "#7B2CBF"])


iris = load_iris()
dados = pd.DataFrame(iris.data, columns=["comprimento_sepala", "largura_sepala", "comprimento_petala", "largura_petala"])
dados["especie"] = [iris.target_names[i] for i in iris.target]
variaveis = ["comprimento_sepala", "largura_sepala", "comprimento_petala", "largura_petala"]
nomes_variaveis = {"comprimento_sepala": "Comprimento da sépala", "largura_sepala": "Largura da sépala",
                   "comprimento_petala": "Comprimento da pétala", "largura_petala": "Largura da pétala"}
especies = list(cores.keys())


print(" estatísticas gerais ")
estatisticas_gerais = pd.DataFrame({
    "media": dados[variaveis].mean(),
    "desvio": dados[variaveis].std(),
    "moda": dados[variaveis].mode().iloc[0],
})
print(estatisticas_gerais.round(3))

print("\n estatísticas por espécie")
for especie in especies:
    subconjunto = dados[dados["especie"] == especie][variaveis]
    estatisticas_especie = pd.DataFrame({
        "media": subconjunto.mean(),
        "desvio": subconjunto.std(),
        "moda": subconjunto.mode().iloc[0],
    })
    print(f"\n{especie}:")
    print(estatisticas_especie.round(3))

print("\n frequência de cada espécie")
print(dados["especie"].value_counts())



def relacao_comprimento_largura(parte):
    coluna_comprimento = f"comprimento_{parte}"
    coluna_largura = f"largura_{parte}"
    nome_parte = {"sepala": "sépala", "petala": "pétala"}[parte]
    print(f"\n=== Correlação comprimento x largura da {parte} ===")
    plt.figure(figsize=(6, 5))
    for especie in especies:
        subconjunto = dados[dados["especie"] == especie]
        correlacao = subconjunto[coluna_comprimento].corr(subconjunto[coluna_largura])
        print(f"{especie}: {correlacao:.3f}")
        plt.scatter(subconjunto[coluna_comprimento], subconjunto[coluna_largura],
                    color=cores[especie], label=f"{especie} (r={correlacao:.2f})", alpha=0.8)
    plt.xlabel(f"Comprimento da {nome_parte} (cm)")
    plt.ylabel(f"Largura da {nome_parte} (cm)")
    plt.title(f"Comprimento × largura da {nome_parte}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"relacao_{parte}.png", dpi=120)


relacao_comprimento_largura("sepala")
relacao_comprimento_largura("petala")


figura, eixos = plt.subplots(2, 2, figsize=(10, 8))
for eixo, variavel in zip(eixos.flat, variaveis):
    for especie in especies:
        eixo.hist(dados[dados["especie"] == especie][variavel], bins=15,
                  color=cores[especie], alpha=0.6, label=especie)
    eixo.set_title(nomes_variaveis[variavel])
    eixo.set_xlabel("cm")
eixos[0, 0].legend()
plt.tight_layout()
plt.savefig("distribuicao_variaveis.png", dpi=120)


dados_padronizados = StandardScaler().fit_transform(dados[variaveis])
pca = PCA(n_components=2)
componentes = pca.fit_transform(dados_padronizados)
print("\n PCA ")
print(f"Variância explicada: PC1={pca.explained_variance_ratio_[0]:.3f}, "
      f"PC2={pca.explained_variance_ratio_[1]:.3f}, total={pca.explained_variance_ratio_.sum():.3f}")
print("Peso de cada variável nos componentes:")
print(pd.DataFrame(pca.components_, columns=variaveis, index=["PC1", "PC2"]).round(3))

plt.figure(figsize=(6, 5))
for especie in especies:
    mascara = dados["especie"] == especie
    plt.scatter(componentes[mascara, 0], componentes[mascara, 1], color=cores[especie], label=especie, alpha=0.8)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("Iris projetado nos 2 primeiros componentes principais")
plt.legend()
plt.tight_layout()
plt.savefig("pca.png", dpi=120)


X = dados[variaveis]
y = dados["especie"]
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
normalizador = StandardScaler().fit(X_treino)
X_treino = normalizador.transform(X_treino)
X_teste = normalizador.transform(X_teste)


valores_k = range(1, 21)
acuracias = []
for k in valores_k:
    modelo = KNeighborsClassifier(n_neighbors=k)
    acuracias.append(cross_val_score(modelo, X_treino, y_treino, cv=5).mean())

melhor_k = valores_k[acuracias.index(max(acuracias))]
print(f"\n=== K-NN ===")
print(f"Melhor k na validação cruzada: {melhor_k} (acurácia média {max(acuracias):.3f})")

plt.figure(figsize=(6, 4))
plt.plot(list(valores_k), acuracias, marker="o", color="#7B2CBF")
plt.axvline(melhor_k, color="#D62828", linestyle="--", label=f"melhor k = {melhor_k}")
plt.xlabel("k (número de vizinhos)")
plt.ylabel("Acurácia média (validação cruzada)")
plt.title("Acurácia do K-NN por valor de k")
plt.legend()
plt.tight_layout()
plt.savefig("knn_acuracia_por_k.png", dpi=120)

modelo_final = KNeighborsClassifier(n_neighbors=melhor_k).fit(X_treino, y_treino)
previsoes = modelo_final.predict(X_teste)
print(f"Acurácia no teste: {accuracy_score(y_teste, previsoes):.3f}")
print(classification_report(y_teste, previsoes))

matriz = confusion_matrix(y_teste, previsoes, labels=especies)
ConfusionMatrixDisplay(matriz, display_labels=especies).plot(cmap=mapa_cores)
plt.title(f"Matriz de confusão (k={melhor_k})")
plt.xlabel("Espécie prevista")
plt.ylabel("Espécie real")
plt.tight_layout()
plt.savefig("knn_matriz_confusao.png", dpi=120)


plt.show()
