static void expr_error(const char *fmt)

{

    term_printf(fmt);

    term_printf("\n");

    longjmp(expr_env, 1);

}
