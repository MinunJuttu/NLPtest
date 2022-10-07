# NLPtest
Добрый день! Начну с козырей: я не выполнил ни одного задания из тестового дуэта. Пытался сделать первое. В предисловии к нему прозвучало: «В любом случае отправляйте любые наработки, даже если не получится выполнить задание полностью.» — поэтому буду рад рассказать о своих потугах. 
ПРИМЕЧАНИЕ: файл testfortest.txt — это «Детство» Толстого.

Начну с хорошего. Я сумел понять, что есть N-граммы и для чего они могут использоваться. Например, для выдачи подходящих сообщений при наборе SMS на смартфоне. С помощью библиотеки NLTK у меня получилось создать набор триграмм. Код получился небольшим, наблюдать его можно в файле NGRAMSnltk.py. Там только с помощью ReGex и NLTK создаётся и сразу выводится (без сохранения) набор триграмм. 

Продолжу не таким бодрым повествованием. 
    Для начала хочу обратить внимание на файлы preparedoc.py и learning.py. В первом код для подготовки документа к работе с библиотекой sklearn. С сожалением обнаружил, что не могу заставить её работать с чем-то, кроме файла из папки, который она считывает как raw (text/dictionary.txt). Как понимаю, это вызвано направленностью библиотеки на работу с корпусами состоящими из множества файлов, а не одним документом. 
    В ходе работы второго документа файл проходит обрабатку для работы с sklearn, проходит обучение через RandomForest и сохраняется с помощью pickle. Результат можно видеть в файле text_classifier. Моя проблема в том, что дальше я не сумел найти способ работы с этой сохранённой моделью.
    
В файлах failed_test ONE.py и failed_test TWO.py можно наблюдать мои потуги работы с методом sklearn.neighbors. Я планировал создать обучить с его помощью модель и затем запрашивать выдачу через input(). Это уже творческая попытка создать велосипед с костылями вместо колёс.

На этом всё. Спасибо за тестовые и ваше время. Мне всё понравилось.
