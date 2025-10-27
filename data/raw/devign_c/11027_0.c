static void sigill_handler (int sig)

{

    if (!canjump) {

        signal (sig, SIG_DFL);

        raise (sig);

    }



    canjump = 0;

    siglongjmp (jmpbuf, 1);

}
