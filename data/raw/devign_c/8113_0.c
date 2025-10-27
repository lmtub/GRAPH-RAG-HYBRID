static int cmp_func_names(const char *a, const char *b)

{

    int ascii_diff, digit_diff;



    for (; !(ascii_diff = *a - *b) && *a; a++, b++);

    for (; av_isdigit(*a) && av_isdigit(*b); a++, b++);



    return (digit_diff = av_isdigit(*a) - av_isdigit(*b)) ? digit_diff : ascii_diff;

}
