static void handle_arg_cpu(const char *arg)

{

    cpu_model = strdup(arg);

    if (cpu_model == NULL || strcmp(cpu_model, "?") == 0) {

        /* XXX: implement xxx_cpu_list for targets that still miss it */

#if defined(cpu_list_id)

        cpu_list_id(stdout, &fprintf, "");

#elif defined(cpu_list)

        cpu_list(stdout, &fprintf); /* deprecated */

#endif

        exit(1);

    }

}
