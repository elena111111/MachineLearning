# Восстановление регрессии

При *Y = R* задача обучения по прецедентам называется *задачей восстановления регрессии*.
Требуется построить функцию регрессии 

<a href="https://www.codecogs.com/eqnedit.php?latex=a:&space;X&space;\rightarrow&space;Y," target="_blank"><img src="https://latex.codecogs.com/gif.latex?a:&space;X&space;\rightarrow&space;Y," title="a: X \rightarrow Y," /></a> где заданы *X* - множество объектов, *Y* - множество ответов.

*Функционал качества (SSE)* определяется как сумма квадратов ошибок:

<a href="https://www.codecogs.com/eqnedit.php?latex=Q(\alpha,&space;X^l)&space;=&space;\sum_{i&space;=&space;1}^{l}(g(x,&space;\alpha)&space;-&space;y_i)^2," target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q(\alpha,&space;X^l)&space;=&space;\sum_{i&space;=&space;1}^{l}(g(x,&space;\alpha)&space;-&space;y_i)^2," title="Q(\alpha, X^l) = \sum_{i = 1}^{l}(g(x, \alpha) - y_i)^2," /></a>

где *g* - модель регрессии (параметрическое семейство функций), 
<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha" title="\alpha" /></a> - вектор параметров модели.

По методу наименьших квадратов найдем вектор параметров, при котором *SSE* минимален. Для этого производную функционала качества по вектору параметров приравняем к нулю.

 # Формула Надарая-Ватсона

Возьмем в качестве модели регрессии константу: 

<a href="https://www.codecogs.com/eqnedit.php?latex=a(x)&space;=&space;g(x,&space;\alpha)&space;=&space;\alpha,&space;\alpha&space;\in&space;R." target="_blank"><img src="https://latex.codecogs.com/gif.latex?a(x)&space;=&space;g(x,&space;\alpha)&space;=&space;\alpha,&space;\alpha&space;\in&space;R." title="a(x) = g(x, \alpha) = \alpha, \alpha \in R." /></a>

Значение *a(x)* вычисляется для каждого *x* по нескольким ближайшим к нему объектам, при этом вводятся веса объектов *w(x)* (зависят от того *x*, в котором собираемся вычислять alpha).
Веса задаются так, чтобы они убывали с увеличением расстояния от *x* до остальных объектов. Для этого вводим функцию ядра (невозрастающую, гладкую, ограниченную):

<a href="https://www.codecogs.com/eqnedit.php?latex=w_i&space;=&space;K&space;\left&space;(\frac{\rho&space;(x,&space;x_i)}{h}&space;\right&space;)." target="_blank"><img src="https://latex.codecogs.com/gif.latex?w_i&space;=&space;K&space;\left&space;(\frac{\rho&space;(x,&space;x_i)}{h}&space;\right&space;)." title="w_i = K \left (\frac{\rho (x, x_i)}{h} \right )." /></a>

*h* называется *шириной окна*.

Решаем задачу методом наименьших квадратов.

<a href="https://www.codecogs.com/eqnedit.php?latex=Q(\alpha,&space;X^l)&space;=&space;\sum_{i&space;=&space;1}^{l}w_i&space;(\alpha&space;-&space;y_i)^2&space;\rightarrow&space;\min\limits_{\alpha}," target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q(\alpha,&space;X^l)&space;=&space;\sum_{i&space;=&space;1}^{l}w_i&space;(\alpha&space;-&space;y_i)^2&space;\rightarrow&space;\min\limits_{\alpha}," title="Q(\alpha, X^l) = \sum_{i = 1}^{l}w_i (\alpha - y_i)^2 \rightarrow \min\limits_{\alpha}," /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;Q}{\partial&space;\alpha}&space;=&space;0." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;Q}{\partial&space;\alpha}&space;=&space;0." title="\frac{\partial Q}{\partial \alpha} = 0." /></a>

Получим *формулу Надарая-Ватсона:*

<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha_h&space;(x,&space;X^l)&space;=&space;\frac{\sum\limits_{i&space;=&space;1}^{l}&space;y_i&space;w_i(x)}{\sum\limits_{i&space;=&space;1}^{l}&space;w_i(x)}." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha_h&space;(x,&space;X^l)&space;=&space;\frac{\sum\limits_{i&space;=&space;1}^{l}&space;y_i&space;w_i(x)}{\sum\limits_{i&space;=&space;1}^{l}&space;w_i(x)}." title="\alpha_h (x, X^l) = \frac{\sum\limits_{i = 1}^{l} y_i w_i(x)}{\sum\limits_{i = 1}^{l} w_i(x)}." /></a>

Чтобы подобрать оптимальное *h*, воспользуемся скользящим контролем с исключением объектов по одному ( *LOO* ):

<a href="https://www.codecogs.com/eqnedit.php?latex=LOO(h,&space;X^l)&space;=&space;\sum_{i&space;=&space;1}^{l}(\alpha_h(x_i,&space;\{X^l\backslash&space;x_i\})&space;-&space;y_i)^2&space;\rightarrow&space;\min\limits_h&space;." target="_blank"><img src="https://latex.codecogs.com/gif.latex?LOO(h,&space;X^l)&space;=&space;\sum_{i&space;=&space;1}^{l}(\alpha_h(x_i,&space;\{X^l\backslash&space;x_i\})&space;-&space;y_i)^2&space;\rightarrow&space;\min\limits_h&space;." title="LOO(h, X^l) = \sum_{i = 1}^{l}(\alpha_h(x_i, \{X^l\backslash x_i\}) - y_i)^2 \rightarrow \min\limits_h ." /></a>


Реализация.

Создан класс *NonparamRegression*, полями которого являются обучающая выборка, ширина окна и ядро. 
Ширина окна подбирается с помощью LOO при создании объекта класса.

Для аппроксимации функции регрессии в одной точке испольмуется метод *predict*:

```python
    def predict(self, test_point):
        numerator, denominator = 0, 0
        for i in range(self.X.shape[0]):
            numerator += self.Y[i] * self.ker(self.__e_dist(test_point, self.X[i]) / self.h)
            denominator += self.ker(self.__e_dist(test_point, self.X[i]) / self.h)
        if denominator == 0:
            return 0
        else:
            return numerator / denominator  # alpha
```  

Для тестирования программы использовалась выборка *boston*, где предсказывается стоимость жилья в зависимости от различных характеристик расположения (в нашем примере 5-му признаку соответствует среднее количество комнат в доме).
Использовались квартическое и гауссовское ядра. Сравним их.

<table><tr>
<th>Ядро</th><th>SSE</th><th>h оптимальное</th>
</tr><tr><td>Квартическое</td><td>17411</td><td>0.55</td>
</tr><tr><tr><td>Гауссовское</td><td>17316</td><td>0.2</td>
</tr></table>

![alt text](https://github.com/elena111111/MachineLearning/blob/master/nad_wat.png)

# LOWESS - локально взвешенное сглаживание

Линия регрессии, полученная по формула Надарая-Ватсона, довольно чувствительна к выбросам.
Поэтому добавим для каждого объекта величину ошибки 

<a href="https://www.codecogs.com/eqnedit.php?latex=\varepsilon_i&space;=&space;|a_h(x_i,&space;X^l\backslash\{x_i\})&space;-&space;y_i|," target="_blank"><img src="https://latex.codecogs.com/gif.latex?\varepsilon_i&space;=&space;|\alpha_h(x_i,&space;X^l\backslash\{x_i\})&space;-&space;y_i|," title="\varepsilon_i = |\alpha_h(x_i, X^l\backslash\{x_i\}) - y_i|," /></a>

и коэффициент

<a href="https://www.codecogs.com/eqnedit.php?latex=\gamma&space;_i&space;=&space;\tilde{K}(|a_i&space;-&space;y_i|)." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\gamma&space;_i&space;=&space;\tilde{K}(|a_i&space;-&space;y_i|)." title="\gamma _i = \tilde{K}(|a_i - y_i|)." /></a>

Чем больше величина ошибки на *i*-м объекте, тем меньше должен быть его вес. Поэтому домножим веса <a href="https://www.codecogs.com/eqnedit.php?latex=w_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w_i" title="w_i" /></a> 
на коэффициенты <a href="https://www.codecogs.com/eqnedit.php?latex=\gamma_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\gamma_i" title="\gamma_i" /></a> .

Алгоритм нахождения коэффициентов работает по следующему принципу:  
1) инициализируем все <a href="https://www.codecogs.com/eqnedit.php?latex=\gamma_i&space;=&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\gamma_i&space;=&space;1" title="\gamma_i = 1" /></a>  
2) до тех пор, пока <a href="https://www.codecogs.com/eqnedit.php?latex=\gamma_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\gamma_i" title="\gamma_i" /></a> не стабилизируются,  
вычислить *loo* и изменить *gamma* 

<a href="https://www.codecogs.com/eqnedit.php?latex=a_i&space;=&space;a_h(x_i,&space;X^l\backslash\{x_i\})&space;=&space;\frac{\sum\limits_{i&space;=&space;1,&space;i&space;\neq&space;j}^{l}&space;y_j&space;\gamma_j&space;K(\frac{\rho&space;(x_i,&space;x_j))}{h})}{\sum\limits_{i&space;=&space;1,&space;i&space;\neq&space;j}^{l}&space;\gamma_j&space;K(\frac{\rho&space;(x_i,&space;x_j))}{h})}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a_i&space;=&space;a_h(x_i,&space;X^l\backslash\{x_i\})&space;=&space;\frac{\sum\limits_{i&space;=&space;1,&space;i&space;\neq&space;j}^{l}&space;y_j&space;\gamma_j&space;K(\frac{\rho&space;(x_i,&space;x_j))}{h})}{\sum\limits_{i&space;=&space;1,&space;i&space;\neq&space;j}^{l}&space;\gamma_j&space;K(\frac{\rho&space;(x_i,&space;x_j))}{h})}" title="a_i = a_h(x_i, X^l\backslash\{x_i\}) = \frac{\sum\limits_{i = 1, i \neq j}^{l} y_j \gamma_j K(\frac{\rho (x_i, x_j))}{h})}{\sum\limits_{i = 1, i \neq j}^{l} \gamma_j K(\frac{\rho (x_i, x_j))}{h})}" /></a>   
<a href="https://www.codecogs.com/eqnedit.php?latex=\gamma&space;_i&space;=&space;\tilde{K}(|a_i&space;-&space;y_i|)." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\gamma&space;_i&space;=&space;\tilde{K}(|a_i&space;-&space;y_i|)." title="\gamma _i = \tilde{K}(|a_i - y_i|)." /></a>

В нашем случае коэффициенты стабилизированы, когда разность *loo* для соседних *gamma* станет меньше какого-то заданного значения.

Реализация.
Создан класс *NonparamRegression*.  

Конструктор:
```python
__init__(self, X, Y, ker)
```  
В конструкторе подбирается ширина окна *h* и настраивается список коэффициентов *gamma*

Метод для классификации точки:
```python
predict(self, test_point)
```

Для тестирования программы использовалась подвыборка выборки *boston* и квартическое ядро.
Сравним алгоритм с предыдущим.
<table><tr>
<th>Алгоритм</th><th>SSE</th><th>h оптимальное</th>
</tr><tr><td>Над-Ват</td><td>873.187</td><td>0.91</td>
</tr><tr><tr><td>LOWESS</td><td>928.710</td><td>1.18</td>
</tr></table>

На графике мы видим, что линия регрессии стала более гладкой и менее чувствительной к выбросам.
![alt text](https://github.com/elena111111/MachineLearning/blob/master/lowess.png)

# Многомерная линейная регрессия

Пусть каждому объекту соответствует его признаковое описание: <a href="https://www.codecogs.com/eqnedit.php?latex=x&space;=&space;(f_1(x),&space;\dots,&space;f_n(x)),&space;f_j:X&space;\rightarrow&space;\mathbb{R}." target="_blank"><img src="https://latex.codecogs.com/gif.latex?x&space;=&space;(f_1(x),&space;\dots,&space;f_n(x)),&space;f_j:X&space;\rightarrow&space;\mathbb{R}." title="x = (f_1(x), \dots, f_n(x)), f_j:X \rightarrow \mathbb{R}." /></a>
Линейной моделью регрессии называется: 

<a href="https://www.codecogs.com/eqnedit.php?latex=g(x,&space;\alpha)&space;=&space;\sum_{i&space;=&space;1}^{n}&space;\alpha_j&space;f_j(x)." target="_blank"><img src="https://latex.codecogs.com/gif.latex?g(x,&space;\alpha)&space;=&space;\sum_{i&space;=&space;1}^{n}&space;\alpha_j&space;f_j(x)." title="g(x, \alpha) = \sum_{i = 1}^{n} \alpha_j f_j(x)." /></a>

Всего имеется *l* объектов обучающей выборки. Обозначим: 
<a href="https://www.codecogs.com/eqnedit.php?latex=F&space;=&space;(f_j(x_i))_{(l,n)},&space;y&space;=&space;(y_i)_{(l,&space;1)},&space;\alpha&space;=&space;(\alpha_j)_{(n,&space;1)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F&space;=&space;(f_j(x_i))_{(l,n)},&space;y&space;=&space;(y_i)_{(l,&space;1)},&space;\alpha&space;=&space;(\alpha_j)_{(n,&space;1)}" title="F = (f_j(x_i))_{(l,n)}, y = (y_i)_{(l, 1)}, \alpha = (\alpha_j)_{(n, 1)}" /></a>, 
-- матрица объектов-признаков, целевой вектор, вектор параметров соответственно.

Функционал качества в матричном виде:

<a href="https://www.codecogs.com/eqnedit.php?latex=Q(\alpha)&space;=&space;||F&space;\alpha&space;-&space;y||^2," target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q(\alpha)&space;=&space;||F&space;\alpha&space;-&space;y||^2," title="Q(\alpha) = ||F \alpha - y||^2," /></a>

условие минимума:

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;Q}{\partial&space;\alpha}&space;(\alpha)&space;=&space;2&space;F^T&space;(F&space;\alpha&space;-&space;y)&space;=&space;0&space;=>&space;F^TF&space;\alpha&space;=&space;F^Ty&space;=>" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;Q}{\partial&space;\alpha}&space;(\alpha)&space;=&space;2&space;F^T&space;(F&space;\alpha&space;-&space;y)&space;=&space;0&space;=>&space;F^TF&space;\alpha&space;=&space;F^Ty&space;=>" title="\frac{\partial Q}{\partial \alpha} (\alpha) = 2 F^T (F \alpha - y) = 0 => F^TF \alpha = F^Ty =>" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha*&space;=&space;(F^TF)^{-1}F^Ty&space;=&space;F^&plus;y,&space;F^&plus;" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha*&space;=&space;(F^TF)^{-1}F^Ty&space;=&space;F^&plus;y,&space;F^&plus;" title="\alpha* = (F^TF)^{-1}F^Ty = F^+y, F^+" /></a> называется псевдообратной матрицей для *F*.

Сингулярное разложение (*SVD*).

Произвольную *(l, n)* матрицу  ранга *n* можно представить в виде сингулярного разложения: 

<a href="https://www.codecogs.com/eqnedit.php?latex=F&space;=&space;VDU^T," target="_blank"><img src="https://latex.codecogs.com/gif.latex?F&space;=&space;VDU^T," title="F = VDU^T," /></a>

где:  
  <a href="https://www.codecogs.com/eqnedit.php?latex=d_{(n,&space;n)}&space;=&space;diag(\sqrt\lambda_1,&space;\dots,&space;\sqrt\lambda\n),&space;\sqrt\lambda_1,&space;\dots,&space;\sqrt\lambda\n&space;-" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d_{(n,&space;n)}&space;=&space;diag(\sqrt\lambda_1,&space;\dots,&space;\sqrt\lambda\n),&space;\sqrt\lambda_1,&space;\dots,&space;\sqrt\lambda\n&space;-" title="d_{(n, n)} = diag(\sqrt\lambda_1, \dots, \sqrt\lambda\n), \sqrt\lambda_1, \dots, \sqrt\lambda\n -" /></a>
общие собственные ненулевые значения матриц <a href="https://www.codecogs.com/eqnedit.php?latex=FF^T,&space;F^TF;" target="_blank"><img src="https://latex.codecogs.com/gif.latex?FF^T,&space;F^TF;" title="FF^T, F^TF;" /></a>  
  <a href="https://www.codecogs.com/eqnedit.php?latex=V_{(l,&space;n)}&space;=&space;(v_1,&space;\dots,&space;v_n)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?V_{(l,&space;n)}&space;=&space;(v_1,&space;\dots,&space;v_n)" title="V_{(l, n)} = (v_1, \dots, v_n)" /></a>
ортогональна, <a href="https://www.codecogs.com/eqnedit.php?latex=V^TV&space;=&space;I_n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?V^TV&space;=&space;I_n" title="V^TV = I_n" /></a>,
столбцы <a href="https://www.codecogs.com/eqnedit.php?latex=v_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?v_j" title="v_j" /></a> - собственные векторы матрицы
<a href="https://www.codecogs.com/eqnedit.php?latex=FF^T," target="_blank"><img src="https://latex.codecogs.com/gif.latex?FF^T," title="FF^T," /></a> 
соответствующие <a href="https://www.codecogs.com/eqnedit.php?latex=\lambda_1,&space;\dots,&space;\lambda_n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\lambda_1,&space;\dots,&space;\lambda_n" title="\lambda_1, \dots, \lambda_n" /></a>;  
  <a href="https://www.codecogs.com/eqnedit.php?latex=U_{(n,&space;n)}&space;=&space;(u_1,&space;\dots,&space;u_n)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?U_{(n,&space;n)}&space;=&space;(u_1,&space;\dots,&space;u_n)" title="U_{(n, n)} = (u_1, \dots, u_n)" /></a>
ортогональна, <a href="https://www.codecogs.com/eqnedit.php?latex=U^TU&space;=&space;I_n," target="_blank"><img src="https://latex.codecogs.com/gif.latex?U^TU&space;=&space;I_n," title="U^TU = I_n," /></a>
столбцы <a href="https://www.codecogs.com/eqnedit.php?latex=u_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?u_j" title="u_j" /></a> - собственные векторы матрицы
<a href="https://www.codecogs.com/eqnedit.php?latex=F^TF," target="_blank"><img src="https://latex.codecogs.com/gif.latex?F^TF," title="F^TF," /></a>
соответствующие <a href="https://www.codecogs.com/eqnedit.php?latex=\lambda_1,&space;\dots,&space;\lambda_n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\lambda_1,&space;\dots,&space;\lambda_n" title="\lambda_1, \dots, \lambda_n" /></a>.

Перепишем псевдообратную матрицу:
<a href="https://www.codecogs.com/eqnedit.php?latex=F^&plus;&space;=&space;(UDV^TVDU^T)^{-1}UDV^T&space;=&space;UD^{-1}V^T" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F^&plus;&space;=&space;(UDV^TVDU^T)^{-1}UDV^T&space;=&space;UD^{-1}V^T" title="F^+ = (UDV^TVDU^T)^{-1}UDV^T = UD^{-1}V^T" /></a>.
SVD используют, чтобы не обращать матрицы.

Реализация.

Создан класс *LinearRegression*, которому в конструкторе передается обучающая выборка, по которой строится вектор парамеров *alpha*:

*Обычная линейная регрессия:*  
```python
def __init__(self, X, Y, typeLR):
    self.alpha = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), Y)
```
*Линейная регрессия + SVD:*  
```python
def __init__(self, X, Y, typeLR):
    self.V, self.D, self.U = np.linalg.svd(self.X, full_matrices=False)
    F_pseudo_inverse = np.dot(np.dot(self.U, np.linalg.inv(np.diag(self.D))), np.transpose(self.V))
    self.alpha = np.dot(F_pseudo_inverse, self.Y)
```

Метод для классификации точки (одинаковый в обоих случаях):  
```python
    def predict(self, test_point):
        return np.dot(self.alpha, np.transpose(test_point))
```

Для тестирования используем выборку *diabetes*.
Результат работы:

Полученная функция регрессии (для всех признаков):  
alpha * test_point = [ 152.13348416  -10.01219782 -239.81908937  519.83978679  324.39042769
 -792.18416163  476.74583782  101.04457032  177.06417623  751.27932109
   67.62538639] * test_point

SSE (для всех признаков): 1263983.1562554843

Теперь выберем 2 признака и визуализируем результат (для алгоритмов с SVD и без в данном случае он одинаковый). Коэффициенты и SSE можно посмотреть ниже в таблице.

![alt text](https://github.com/elena111111/MachineLearning/blob/master/LR_diabetes_3_6a.png)

# Гребневая регрессия

Реализация. 

Создан класс *LinearRegression*, которому в конструкторе передается обучающая выборка, и из конструктора вызывается метод *find_tau*, и затем *find_alpha*:

Подбор параметра *tau* по *LOO*:

```python
    def _find_tau(self):
        min_loo = sys.maxsize
        best_tau = 0.1
        tau = np.arange(0.0001, 0.9999, 0.005)
        for t in tau:
            self.tau = t
            self.alpha = self._find_alpha()
            cur_loo = self._loo()
            if cur_loo < min_loo:
                min_loo = cur_loo
                best_tau = self.tau
        self.tau = best_tau
```

*Обычная гребневая регрессия:*

```python
def _find_alpha(self):
    I = np.eye(self.X.shape[1])
    A = np.linalg.inv(np.dot(np.transpose(self.X), self.X) + np.dot(self.tau, I))
    self.alpha =  np.dot(np.dot(A, np.transpose(self.X)), self.Y)
```
*Гребневая регрессия + SVD:*

```python
def _find_alpha(self):
    D1 = np.diag(self.D)
    I = np.eye(self.X.shape[1])
    A = np.linalg.inv(np.dot(D1, D1) + np.dot(self.tau, I))
    B = np.dot(np.dot(np.dot(self.U, A), np.diag(self.D)), np.transpose(self.V))
    self.alpha = np.dot(B, self.Y)
```

Рассмотрим выборку *boston* по 4 и 6 признакам (каждый ее столбец нормируем и центрируем, чтобы рассматривать *tau* от 0 до 1).
Она имеет большое число обусловленности (175).

Подбор *tau*:

![alt text](https://github.com/elena111111/MachineLearning/blob/master/loo_for_RR_boston_4_6.png)

![alt text](https://github.com/elena111111/MachineLearning/blob/master/loo_for_RR_SVD_boston_4_6.png)
 
Как работает гребневая регрессия:

![alt text](https://github.com/elena111111/MachineLearning/blob/master/RR_boston_4_6.png)

![alt text](https://github.com/elena111111/MachineLearning/blob/master/RR_SVD_boston_4_6.png)

И для сравнения, линейная регрессия из предыдущего случая (SVD):

![alt text](https://github.com/elena111111/MachineLearning/blob/master/LR_SVD_boston_4_6.png)

Коэффициенты и SSE в таблице ниже.

Мы видим, что гребневая регрессия действительно помогла решить проблему мультиколлинеарности и сделала коэффициенты более устойчивыми (это видно в примере с SVD).


Сравнение алгоритмов:
<table><tr>
<th>Алгоритм</th><th>Выборка, признаки, число обусловленности</th><th>SSE</th><th>*alpha*</th><th>*tau*</th><th>loo</th>
</tr><tr><td>Линейная регрессия</td><td>*diabetes*, 3, 6, 23.19</td><td>1840014.4178414412</td><td>[ 152.13348416  620.30960865 -528.25798488]</td><td>--</td><td>--</td>
</tr><tr><td>Линейная регрессия + SVD</td><td>*diabetes*, 3, 6, 23.19</td><td>1840014.4178414424</td><td>[ 152.13348416  620.30960865 -528.25798488]</td><td>--</td><td>--</td>
</tr><tr><td>Линейная регрессия</td><td>*boston*, 4, 6, 175.05</td><td>34535.35187773189</td><td>[  22.53280632 -329.83800805  -75.42940867]</td><td>--</td><td>--</td>
</tr><tr><td>Линейная регрессия + SVD</td><td>*boston*, 4, 6, 175.05</td><td>85359.85219729338</td><td>[ 22.53280632 164.81724712 295.49616292]</td><td>--</td><td>--</td>
</tr><tr><td>Гребневая регрессия</td><td>*boston*, 4, 6, 175.05</td><td>34535.39815610829</td><td>[  22.53280187 -328.30043121  -75.98221535]</td><td>0.0001</td><td>34886.79619275586</td>
</tr><tr><td>Гребневая регрессия + SVD</td><td>*boston*, 4, 6, 175.05</td><td>44225.39054893582</td><td>[ 22.48858026 -7.9524389  28.74538325]</td><td>0.9951</td><td>44373.737437263015</td>
</tr></table>

# Метод главных компонент (PCA)

Используется для решения проблемы мультиколлинеарности. Используется, чтобы преобразовать исходные признаки, и получить новые признаки, гаратировав их линейную независимость, и возможно уменьшив размерность задачи.
Также может быть удобен для наглядной визуализации данных в 3х-мерном пространстве.

В методе главных компонент строится минимальное число новых признаков, по которым исходные признаки восстанавливаются линейным преобразованием с минимальными погрешностями.
Относится к методам обучения без учителя.

Пусть объекты задаются их признаковым описанием:
<a href="https://www.codecogs.com/eqnedit.php?latex=x_i&space;=&space;(f_1(x_i),&space;\dots,&space;f_n(x_i)),&space;i&space;=&space;\overline{1,&space;l}." target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_i&space;=&space;(f_1(x_i),&space;\dots,&space;f_n(x_i)),&space;i&space;=&space;\overline{1,&space;l}." title="x_i = (f_1(x_i), \dots, f_n(x_i)), i = \overline{1, l}." /></a>
Матрица признаков:  
<a href="https://www.codecogs.com/eqnedit.php?latex=F_{(l,&space;n)}&space;=&space;\bigl(\begin{smallmatrix}&space;f_1(x_1)&space;&&space;\dots&space;&&space;f_n(x_1)\\&space;\dots&space;&&space;\dots&space;&&space;\dots\\&space;f_1(x_l)&space;&&space;\dots&space;&&space;f_n(x_l)&space;\end{smallmatrix}\bigr)&space;=&space;\begin{pmatrix}&space;x_1\\&space;\dots\\&space;x_l&space;\end{pmatrix}&space;." target="_blank"><img src="https://latex.codecogs.com/gif.latex?F_{(l,&space;n)}&space;=&space;\bigl(\begin{smallmatrix}&space;f_1(x_1)&space;&&space;\dots&space;&&space;f_n(x_1)\\&space;\dots&space;&&space;\dots&space;&&space;\dots\\&space;f_1(x_l)&space;&&space;\dots&space;&&space;f_n(x_l)&space;\end{smallmatrix}\bigr)&space;=&space;\begin{pmatrix}&space;x_1\\&space;\dots\\&space;x_l&space;\end{pmatrix}&space;." title="F_{(l, n)} = \bigl(\begin{smallmatrix} f_1(x_1) & \dots & f_n(x_1)\\ \dots & \dots & \dots\\ f_1(x_l) & \dots & f_n(x_l) \end{smallmatrix}\bigr) = \begin{pmatrix} x_1\\ \dots\\ x_l \end{pmatrix} ." /></a>

Матрица признаков в новом пространстве, *m < n*:  
<a href="https://www.codecogs.com/eqnedit.php?latex=G_{(l,&space;m)}&space;=&space;\bigl(\begin{smallmatrix}&space;g_1(x_1)&space;&&space;\dots&space;&&space;g_m(x_1)\\&space;\dots&space;&&space;\dots&space;&&space;\dots\\&space;g_1(x_l)&space;&&space;\dots&space;&&space;g_m(x_l)&space;\end{smallmatrix}\bigr)&space;=&space;\begin{pmatrix}&space;z_1\\&space;\dots\\&space;z_l&space;\end{pmatrix}&space;." target="_blank"><img src="https://latex.codecogs.com/gif.latex?G_{(l,&space;m)}&space;=&space;\bigl(\begin{smallmatrix}&space;g_1(x_1)&space;&&space;\dots&space;&&space;g_m(x_1)\\&space;\dots&space;&&space;\dots&space;&&space;\dots\\&space;g_1(x_l)&space;&&space;\dots&space;&&space;g_m(x_l)&space;\end{smallmatrix}\bigr)&space;=&space;\begin{pmatrix}&space;z_1\\&space;\dots\\&space;z_l&space;\end{pmatrix}&space;." title="G_{(l, m)} = \bigl(\begin{smallmatrix} g_1(x_1) & \dots & g_m(x_1)\\ \dots & \dots & \dots\\ g_1(x_l) & \dots & g_m(x_l) \end{smallmatrix}\bigr) = \begin{pmatrix} z_1\\ \dots\\ z_l \end{pmatrix} ." /></a>

Требуем, чтобы исходные признаковые описания можно было восстановить с помощью матрицы линейного преобразования <a href="https://www.codecogs.com/eqnedit.php?latex=U_{(n,&space;m)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?U_{(n,&space;m)}" title="U_{(n, m)}" /></a>:  
<a href="https://www.codecogs.com/eqnedit.php?latex=\widehat{f_j(x)}&space;=&space;\sum_{s&space;=&space;1}^{m}g_s(x)&space;u_{js},&space;j&space;=&space;\overline{1,&space;n}." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\widehat{f_j(x)}&space;=&space;\sum_{s&space;=&space;1}^{m}g_s(x)&space;u_{js},&space;j&space;=&space;\overline{1,&space;n}." title="\widehat{f_j(x)} = \sum_{s = 1}^{m}g_s(x) u_{js}, j = \overline{1, n}." /></a> Или <a href="https://www.codecogs.com/eqnedit.php?latex=\widehat{x}&space;=&space;zU^T" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\widehat{x}&space;=&space;zU^T" title="\widehat{x} = zU^T" /></a>.

Число главных компонент *m* (эффективную размерность задачи) можно найти так:  
упорядочить поо убыванию собственные числа матрицы 
<a href="https://www.codecogs.com/eqnedit.php?latex=F^TF:&space;\lambda_1&space;\geqslant&space;\dots&space;\geqslant&space;\lambda_n," target="_blank"><img src="https://latex.codecogs.com/gif.latex?F^TF:&space;\lambda_1&space;\geqslant&space;\dots&space;\geqslant&space;\lambda_n," title="F^TF: \lambda_1 \geqslant \dots \geqslant \lambda_n," /></a>
задать число <a href="https://www.codecogs.com/eqnedit.php?latex=\epsilon&space;\in&space;[0,&space;1]," target="_blank"><img src="https://latex.codecogs.com/gif.latex?\epsilon&space;\in&space;[0,&space;1]," title="\epsilon \in [0, 1]," /></a>
найти наименьшее целое *m*, удовлетворяющее условию:  

<a href="https://www.codecogs.com/eqnedit.php?latex=E(m)&space;=&space;\frac{\lambda_{m&plus;1}&space;&plus;&space;\dots&space;&plus;&space;\lambda_n}{\lambda_{1}&space;&plus;&space;\dots&space;&plus;&space;\lambda_n}&space;\leqslant&space;\epsilon." target="_blank"><img src="https://latex.codecogs.com/gif.latex?E(m)&space;=&space;\frac{\lambda_{m&plus;1}&space;&plus;&space;\dots&space;&plus;&space;\lambda_n}{\lambda_{1}&space;&plus;&space;\dots&space;&plus;&space;\lambda_n}&space;\leqslant&space;\epsilon." title="E(m) = \frac{\lambda_{m+1} + \dots + \lambda_n}{\lambda_{1} + \dots + \lambda_n} \leqslant \epsilon." /></a>  

Величина *E(m)* показывает, какая доля информации теряется при замене признаковых описаний на более короткие. При поиске *m* можно также прибегать к критерию "крутого обрыва": 
выбирать то *m*, где *E(m - 1) >> E(M)*.

Реализация.

Конструктор (вход - матрица признаков):
```python
    def __init__(self, X):
        self.X = X
        v, w = la.eig(np.dot(np.transpose(self.X), self.X)) #собств знач и векторы
        self.Eigens = np.column_stack((np.asarray(v), np.asarray(w)))
        self.Eigens = np.asarray(sorted(self.Eigens, key=lambda x: x[0], reverse=True))
        self._find_m()
        self.U = self.Eigens[:, 1:(self.m + 1)]
        self.G = np.dot(self.X, self.U)
```

Подбор эффективной размерности:  
```python
    def _find_m(self):
        eig_val = self.Eigens[:, 0]
        eps = np.arange(0., 0.9999, 0.005)
        denom = sum(eig_val)
        best_Em, best_Eps = 10, 10
        for e in eps:
            m, Em = 1, e + 1.
            Em1 = Em
            while Em > e:
                numer = sum(eig_val[(m + 1):])
                Em = numer / denom
                if Em < best_Em:
                    best_Eps, best_Em = e, Em
                m += 1
                Em1 = Em
```

Пример работы:

![alt text](https://github.com/elena111111/MachineLearning/blob/master/pca_m_boston.png)

В данном случае можно сократить количество признаков до 2 или 3.

# Нелинейная модель регрессии

Реализация.

Создан класс *NonlinearRegression*, конструктор:  
```python
def __init__(self, X, Y, f)
```
X, Y - обучающая выборка, f - функция (модель). Еще задается шаг *h*, начальное приближение для *alpha* и подбираем *alpha*:  
```python
    def _find_alpha(self):
        eps = 0.001
        cur_Q = self._Q(self.alpha) + eps + 10
        while abs(cur_Q - self._Q(self.alpha)) > eps:
            print("alpha: " + str(self.alpha))
            print("Q: " + str(cur_Q))
            self._find_grad()
            self._find_hessian_inv()
            cur_Q = self._Q(self.alpha)
            self.alpha = self.alpha - np.dot(np.dot(self.h, self.hessian_inv), self.grad)
        return self.alpha
```
Визуализация:

# Картинка