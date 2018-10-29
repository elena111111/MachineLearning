# �������������� ���������

��� *Y = R* ������ �������� �� ����������� ���������� *������� �������������� ���������*.
��������� ��������� ������� ��������� 

<a href="https://www.codecogs.com/eqnedit.php?latex=a:&space;X&space;\rightarrow&space;Y," target="_blank"><img src="https://latex.codecogs.com/gif.latex?a:&space;X&space;\rightarrow&space;Y," title="a: X \rightarrow Y," /></a>

������ *X - * ��������� ��������, *Y - * ��������� �������.

*���������� �������� (SSE)* ������������ ��� ����� ��������� ������:

<a href="https://www.codecogs.com/eqnedit.php?latex=Q(\alpha,&space;X^l)&space;=&space;\sum_{i&space;=&space;1}^{l}(g(x,&space;\alpha)&space;-&space;y_i)^2," target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q(\alpha,&space;X^l)&space;=&space;\sum_{i&space;=&space;1}^{l}(g(x,&space;\alpha)&space;-&space;y_i)^2," title="Q(\alpha, X^l) = \sum_{i = 1}^{l}(g(x, \alpha) - y_i)^2," /></a>

��� *g - * ������ ��������� (��������������� ��������� �������), 
<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha" title="\alpha" /></a> - ������ ���������� ������.

�� ������ ���������� ��������� ������ ������ ����������, ��� ������� *SSE* ���������. ��� ����� ����������� ����������� �������� �� ������� ���������� ���������� � ����.

 # ������� �������-�������

������� � �������� ������ ��������� ���������: 

<a href="https://www.codecogs.com/eqnedit.php?latex=a(x)&space;=&space;g(x,&space;\alpha)&space;=&space;\alpha,&space;\alpha&space;\in&space;R." target="_blank"><img src="https://latex.codecogs.com/gif.latex?a(x)&space;=&space;g(x,&space;\alpha)&space;=&space;\alpha,&space;\alpha&space;\in&space;R." title="a(x) = g(x, \alpha) = \alpha, \alpha \in R." /></a>

�������� *a(x)* ����������� ��� ������� *x* �� ���������� ��������� � ���� ��������, ��� ���� �������� ���� �������� *w(x)* (������� �� ���� *x*, � ������� ���������� ��������� alpha).
���� �������� ���, ����� ��� ������� � ����������� ���������� �� *x* �� ��������� ��������. ��� ����� ������ ������� ���� (��������������, �������, ������������):

<a href="https://www.codecogs.com/eqnedit.php?latex=w_i&space;=&space;K&space;\left&space;(\frac{\rho&space;(x,&space;x_i)}{h}&space;\right&space;)." target="_blank"><img src="https://latex.codecogs.com/gif.latex?w_i&space;=&space;K&space;\left&space;(\frac{\rho&space;(x,&space;x_i)}{h}&space;\right&space;)." title="w_i = K \left (\frac{\rho (x, x_i)}{h} \right )." /></a>

*h* ���������� *������� ����*.

������ ������ ������� ���������� ���������.

<a href="https://www.codecogs.com/eqnedit.php?latex=Q(\alpha,&space;X^l)&space;=&space;\sum_{i&space;=&space;1}^{l}w_i&space;(\alpha&space;-&space;y_i)^2&space;\rightarrow&space;\min\limits_{\alpha}," target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q(\alpha,&space;X^l)&space;=&space;\sum_{i&space;=&space;1}^{l}w_i&space;(\alpha&space;-&space;y_i)^2&space;\rightarrow&space;\min\limits_{\alpha}," title="Q(\alpha, X^l) = \sum_{i = 1}^{l}w_i (\alpha - y_i)^2 \rightarrow \min\limits_{\alpha}," /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;Q}{\partial&space;\alpha}&space;=&space;0." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;Q}{\partial&space;\alpha}&space;=&space;0." title="\frac{\partial Q}{\partial \alpha} = 0." /></a>

������� *������� �������-�������:*

<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha_h&space;(x,&space;X^l)&space;=&space;\frac{\sum\limits_{i&space;=&space;1}^{l}&space;y_i&space;w_i(x)}{\sum\limits_{i&space;=&space;1}^{l}&space;w_i(x)}." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha_h&space;(x,&space;X^l)&space;=&space;\frac{\sum\limits_{i&space;=&space;1}^{l}&space;y_i&space;w_i(x)}{\sum\limits_{i&space;=&space;1}^{l}&space;w_i(x)}." title="\alpha_h (x, X^l) = \frac{\sum\limits_{i = 1}^{l} y_i w_i(x)}{\sum\limits_{i = 1}^{l} w_i(x)}." /></a>

����� ��������� ����������� *h*, ������������� ���������� ��������� � ����������� �������� �� ������ ( *LOO* ):

<a href="https://www.codecogs.com/eqnedit.php?latex=LOO(h,&space;X^l)&space;=&space;\sum_{i&space;=&space;1}^{l}(\alpha_h(x_i,&space;\{X^l\backslash&space;x_i\})&space;-&space;y_i)^2&space;\rightarrow&space;\min\limits_h&space;." target="_blank"><img src="https://latex.codecogs.com/gif.latex?LOO(h,&space;X^l)&space;=&space;\sum_{i&space;=&space;1}^{l}(\alpha_h(x_i,&space;\{X^l\backslash&space;x_i\})&space;-&space;y_i)^2&space;\rightarrow&space;\min\limits_h&space;." title="LOO(h, X^l) = \sum_{i = 1}^{l}(\alpha_h(x_i, \{X^l\backslash x_i\}) - y_i)^2 \rightarrow \min\limits_h ." /></a>


����������.

������ ����� *NonparamRegression*, ������ �������� �������� ��������� �������, ������ ���� � ����. 
������ ���� ����������� � ������� LOO ��� �������� ������� ������.

��� ������������� ������� ��������� � ����� ����� ������������ ����� *predict*:

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

��� ������������ ��������� �������������� ������� *boston*, ��� ��������������� ��������� ����� � ����������� �� ��������� ������������� ������������.
�������������� ������������ � ����������� ����. ������� ��.

<table><tr>
<th>����</th><th>SSE</th><th>h �����������</th>
</tr><tr><td>������������</td><td>17411</td><td>0.55</td>
</tr><tr><tr><td>�����������</td><td>17316</td><td>0.2</td>
</tr></table>

