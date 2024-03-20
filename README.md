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

## от 13.03 (файл 13_3.cpp)

*1. Чтение строк из файла между потоками с использованием директивы omp parallel for. Вывести наиболее часто встречающиеся слова и их количетсво в тексте.*