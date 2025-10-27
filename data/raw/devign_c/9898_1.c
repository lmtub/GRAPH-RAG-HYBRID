static pid_t qtest_qemu_pid(QTestState *s)

{

    FILE *f;

    char buffer[1024];

    pid_t pid = -1;



    f = fopen(s->pid_file, "r");

    if (f) {

        if (fgets(buffer, sizeof(buffer), f)) {

            pid = atoi(buffer);

        }

        fclose(f);

    }

    return pid;

}
