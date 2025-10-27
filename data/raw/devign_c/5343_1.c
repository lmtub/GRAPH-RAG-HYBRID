static int find_debugfs(char *debugfs)

{

    char type[100];

    FILE *fp;



    fp = fopen("/proc/mounts", "r");

    if (fp == NULL) {

        return 0;

    }



    while (fscanf(fp, "%*s %" STR(PATH_MAX) "s %99s %*s %*d %*d\n",

                  debugfs, type) == 2) {

        if (strcmp(type, "debugfs") == 0) {

            break;

        }

    }

    fclose(fp);



    if (strcmp(type, "debugfs") != 0) {

        return 0;

    }

    return 1;

}
