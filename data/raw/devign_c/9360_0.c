int avpriv_vsnprintf(char *restrict s, size_t n, const char *restrict fmt,

                     va_list ap)

{

    int ret;



    if (n == 0)

        return 0;

    else if (n > INT_MAX)

        return AVERROR(EINVAL);



    /* we use n - 1 here because if the buffer is not big enough, the MS

     * runtime libraries don't add a terminating zero at the end. MSDN

     * recommends to provide _snprintf/_vsnprintf() a buffer size that

     * is one less than the actual buffer, and zero it before calling

     * _snprintf/_vsnprintf() to workaround this problem.

     * See http://msdn.microsoft.com/en-us/library/1kt27hek(v=vs.80).aspx */

    memset(s, 0, n);

    ret = vsnprintf(s, n - 1, fmt, ap);

    if (ret == -1)

        ret = n;



    return ret;

}
