static int show_hwaccels(void *optctx, const char *opt, const char *arg)

{

    int i;



    printf("Hardware acceleration methods:\n");

    for (i = 0; hwaccels[i].name; i++) {

        printf("%s\n", hwaccels[i].name);

    }

    printf("\n");

    return 0;

}
