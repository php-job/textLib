# textLib

##### Небольшая конструкция для поиска дублей предложений в тексе

``` python
from textLib import TxtDedubler 
import pandas as pd

df = pd.read_csv('./AbtBuy_Buy.csv')
df = df[['unique_id','title']]
df = df.rename(index=str, columns={'unique_id':'ID','title':'TEXT'})
df = df.reset_index()


ded = TxtDedubler()
df = ded.predict(df, radius=0.3)
df.sort_values('similar_ids_len', ascending=False)
```

На вход принимает датафрейм на выходе тот же датафрейм с дополнительными столбцами  
Входные данные должны содержать обязательные поля **ID** ,**TEXT**  
 
 * * В для поиска дублей используется мешок слов со стемингом и косинусная мера схожести * *

 ### пример таблицы данных
 - аргумент ***radius*** метода ***predict*** - это мера схожести элементов, то есть с каким уровнем схожести считать элементы похожими
 - ***similar_id*** - идентификаторы похожих записей
 
 ![title](https://github.com/php-job/textLib/blob/master/sc1.png)
