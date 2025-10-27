static void do_test_equality(bool expected, int _, ...)
{
    va_list ap_count, ap_extract;
    QObject **args;
    int arg_count = 0;
    int i, j;
    va_start(ap_count, _);
    va_copy(ap_extract, ap_count);
    while (va_arg(ap_count, QObject *) != &test_equality_end_of_arguments) {
        arg_count++;
    }
    va_end(ap_count);
    args = g_new(QObject *, arg_count);
    for (i = 0; i < arg_count; i++) {
        args[i] = va_arg(ap_extract, QObject *);
    }
    va_end(ap_extract);
    for (i = 0; i < arg_count; i++) {
        g_assert(qobject_is_equal(args[i], args[i]) == true);
        for (j = i + 1; j < arg_count; j++) {
            g_assert(qobject_is_equal(args[i], args[j]) == expected);
        }
    }
}