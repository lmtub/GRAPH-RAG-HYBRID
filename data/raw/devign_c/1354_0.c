void checkasm_report(const char *name, ...)

{

    static int prev_checked, prev_failed, max_length;



    if (state.num_checked > prev_checked) {

        print_cpu_name();



        if (*name) {

            int pad_length = max_length;

            va_list arg;



            fprintf(stderr, " - ");

            va_start(arg, name);

            pad_length -= vfprintf(stderr, name, arg);

            va_end(arg);

            fprintf(stderr, "%*c", FFMAX(pad_length, 0) + 2, '[');

        } else

            fprintf(stderr, " - %-*s [", max_length, state.current_func->name);



        if (state.num_failed == prev_failed)

            color_printf(COLOR_GREEN, "OK");

        else

            color_printf(COLOR_RED, "FAILED");

        fprintf(stderr, "]\n");



        prev_checked = state.num_checked;

        prev_failed  = state.num_failed;

    } else if (!state.cpu_flag) {

        int length;



        /* Calculate the amount of padding required to make the output vertically aligned */

        if (*name) {

            va_list arg;

            va_start(arg, name);

            length = vsnprintf(NULL, 0, name, arg);

            va_end(arg);

        } else

            length = strlen(state.current_func->name);



        if (length > max_length)

            max_length = length;

    }

}
