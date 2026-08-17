void square_all(double * data, int n) {
    for (double * ptr = data; ptr < data + n; ptr++) {
        *ptr = *ptr * *ptr;
    }
}