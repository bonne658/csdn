```cpp
#include <iostream>
#include <string>
#include <stdio.h>
#include <time.h>
//#include <sys/time.h>
 
using namespace std;
//统计小于n的素数个数
int prime(int n) {
    //记录某数是否访问
    bool *v = new bool[n];
    //保存素数
    int *p = new int[n];
    int i, j, res = 2;
    if (n < 3) return 0;
    if (n == 3) return 1;
    if (n == 4 || n == 5) return 2;
    for (i = 0; i < n; i++)
        v[i] = false;
    for (i = 1;; i++) {
        //只需判断6n-1和6n+1
        if (6 * i - 1 >= n)
            break;
        if (v[6 * i - 1] == false)
            p[res++] = 6 * i - 1;
        for (j = 2; j < res && (6 * i - 1) * p[j] < n; j++) {
            v[(6 * i - 1) * p[j]] = true;
            if ((6 * i - 1) % p[j] == 0)
                break;
        }
        if (6 * i + 1 >= n)
            break;
        if (v[6 * i + 1] == false)
            p[res++] = 6 * i + 1;
        for (j = 2; j < res && (6 * i + 1) * p[j] < n; j++) {
            v[(6 * i + 1) * p[j]] = true;
            if ((6 * i + 1) % p[j] == 0)
                break;
        }
    }
    delete []v;
    delete []p;
    return res;
}
 
int main() {
    //计算运行时间
    /*
    struct timeval start, end;
    gettimeofday(&start, NULL);
    ...
    gettimeofday(&end, NULL);
    float cost_time = (end.tv_usec-start.tv_usec)/1000000.0 + end.tv_sec-start.tv_sec;
    */
    clock_t s, e;
    s = clock();
    int n = prime(9973);
    e = clock();
    cout << n << "  " << 1.0 * (e - s) / CLOCKS_PER_SEC << endl;
    return 0;
}
```
