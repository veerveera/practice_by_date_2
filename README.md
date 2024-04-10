# Задания с практических занятий 

## Задание по предмету Основы параллельных вычислений
## от 28.02 (файл 28_2.cpp)

*1. Вывести несколько раз приветствие.*

*2. Ввести число для обозначения количетсва потоков. Вывести порядок потоков.*

*3. Ввести число для обозначения количетсва потоков. Вывести потоки по их номерам с помощью omp_get_thread_num().*

*4. Ввести число, ввести количетсво нитей. Каждая нить выводит свою частичную сумму и один раз выводится общая сумма.*

## от 6.03 (файл 6_3.cpp)

*1. Вычисление суммы элементов массива.*

*Написать программу на C++, используя OpenMP, для вычисления суммы элементов массива. Программа должна разделить массив на равные части между потоками и вычислять сумму элементов каждой части. Затем суммы, вычисленные каждым потоком, должны быть сложены для получения общей суммы элементов массива.*

*2. Параллельная сортировка слиянием.*

*Написать программу на C++ с использованием OpenMP для параллельной сортировки массива целых чисел методом слияния. Программа должна разбивать массив на равные части между потоками для эффективного параллельного выполнения сортировки.*

## от 13.03 (файлы 13_3.cpp, dis.txt)

*Чтение строк из файла между потоками с использованием директивы omp parallel for. Вывести наиболее часто встречающиеся слова и их количетсво в тексте.*

## от 20.03 (файлы 20_3.cpp, source_mat.jpg)

*Параллельно с помощью директивы sections в OpenMP преобразования картинки (Поиграться).*

## от 3.04 (файл 3_4.cpp)

*Реализация фрактала Мандельброта с использованием MPI включает в себя несколько ключевых шагов:*

*1. Определение параметров; 2. Разделение задач; 3. Генерация фрактала; 4. Определение принадлежности; 5. Сбор результатов; 6. Визуализация; 7. Завершение.*

## от 10.04 (файл 10_4.cpp, AUZ.mp4)

*Распараллелить задание от 9.04. (https://github.com/veerveera/practice_by__date/blob/main/9_04.cpp)*

*Без параллельности программа работала по времени 239, после за 92.*
