static void slirp_smb_cleanup(SlirpState *s)

{

    char cmd[128];

    int ret;



    if (s->smb_dir[0] != '\0') {

        snprintf(cmd, sizeof(cmd), "rm -rf %s", s->smb_dir);

        ret = system(cmd);

        if (!WIFEXITED(ret)) {

            qemu_error("'%s' failed.\n", cmd);

        } else if (WEXITSTATUS(ret)) {

            qemu_error("'%s' failed. Error code: %d\n",

                    cmd, WEXITSTATUS(ret));

        }

        s->smb_dir[0] = '\0';

    }

}
