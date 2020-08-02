## LSA 主题模型



### 奇异值分解

首先指出: 任意对称矩阵 $A_{n \times n}$ 都可以进行特征值分解. 即: 

$$\begin{aligned} A_{n \times n} = X \Sigma X^{-1} = X \Sigma X^{T} \end{aligned}$$ 

其中: 

* $A$ 是任意 $n$ 阶对称矩阵. 
* $\Sigma$ 是矩阵 $A$ 的特征值组成的对角矩了. 
* $X$ 是由特征向量组成的 $n$ 阶矩阵. 其中每一列是一个特征向量. 



对于任意矩阵 $A_{m \times n}$, 有: $A_{1} = AA^{T}$, $A_{2} = A^{T}A$, 其中 $A_{1}$ 是 $m \times m$ 阶对称矩阵, $A_{2}$ 是 $n \times n$ 阶对称矩阵. 显然 $A_{1}, A_{2}$ 对可以进行特征值分解, 对角化. 有: 

$$\begin{aligned} A_{1} &= AA^{T} = P \Lambda_{1} P^{T} \\ A_{2} &= A^{T}A = Q \Lambda_{2} Q^{T} \end{aligned}$$ 

有: 

$$\begin{aligned} A^{T} = A^{-1} P \Lambda_{1} P^{T} \\ A^{T} = Q \Lambda_{2} Q^{T} A^{-1} \\ A^{-1} P \Lambda_{1} P^{T} = Q \Lambda_{2} Q^{T} A^{-1} \end{aligned}$$ 



$$\begin{aligned} A= P \Sigma Q^{T} = P \Sigma Q^{-1} \end{aligned}$$ 

$$\begin{aligned} AA^{T} &= P \Sigma Q^{T} Q \Sigma^{T} P^{T} \\ &= P \Sigma E \Sigma^{T} P^{T} \\ &= P \Sigma \Sigma^{T} P^{T} \\ &= P \Lambda_{1} P^{T} \end{aligned}$$ 

$$\begin{aligned} A^{T}A &= Q \Sigma^{T} P^{T} P \Sigma Q^{T} \\ &= Q \Sigma^{T} E \Sigma Q^{T} \\ &= Q \Sigma^{T} \Sigma Q^{T} \\ &= Q \Lambda_{2} Q^{T} \end{aligned}$$ 



$\Sigma$ 的求解: 

https://blog.csdn.net/u011251945/article/details/81362642

https://www.cnblogs.com/marsggbo/p/10155801.html





奇异值分解: 

$$\begin{aligned} A= P \Sigma Q^{T} = P \Sigma Q^{-1} \end{aligned}$$ 

其中: 

* $A$ 是 $m \times n$ 的矩阵. 
* $P$ 是 $m \times m$ 的矩阵.
* $\Sigma$ 是 $m \times n$ 的对角矩阵. 一般在 $\Sigma$ 中大于 $n$ 行的部分都为 0. 所以有时候会被表示成 $n \times n$ 的对角矩阵. 
* $Q$ 是 $n \times n$ 的矩阵. 



### LSA 主题模型

在文档主题分析中, 我们可以将奇异值分解的矩阵作如下理解: 

* $m$ 个文档有 $m$ 个主题. $n$ 个词语有 $n$ 个词义. 即: 每个文档由各主题按线性加权表示, 每个词语由各词义按线性加权表示. 

* $A_{m \times n}$ 中, 每行代表一个文档, 每列代表一个词. 矩阵值表示各文档中各词的比例. 
* $P_{m \times m}$ 中, 每行代表一个文档, 每列代表一个主题. 矩阵值表示各文档由各主题按比例混合而成. 
* $Q^{T}_{n \times n}$ 表示将词语向量映射为词义向量. 
* $\Sigma_{m \times n}$ 表示将词义向量映射为题主向量. 
* $\Sigma Q^{T}$ 表示的是, 将词语向量映射到主题向量空间. 值得注意的是: $\Sigma_{m \times n}$ 是特征值矩阵, $Q_{n \times n}^{T}$ 是特征向量矩阵, 词语向量与 $Q_{n \times n}^{T}$ 运算, 是将词语向量映射到词义空间. 而 $\Sigma_{m \times n}$ 是对向量在各维度的缩放, 缩放后的向量被理解为主题向量. $\Sigma Q^{T}$ 的整体运算, 代表的是将词语向量映射为主题向量 (即: 每个词语都有各主题的含义). 



通过以上的推导, 我们可以想到, 当用户需要搜索文档时, 它可以输入一个词语向量(或一组词语, 然后形成词语向量). 通过对词语向量的运算, 我们可以找到最匹配的文档. 





