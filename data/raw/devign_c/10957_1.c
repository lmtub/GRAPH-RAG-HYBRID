print_syscall_ret(int num, abi_long ret)

{

    int i;



    for(i=0;i<nsyscalls;i++)

        if( scnames[i].nr == num ) {

            if( scnames[i].result != NULL ) {

                scnames[i].result(&scnames[i],ret);

            } else {

                if( ret < 0 ) {

                    gemu_log(" = -1 errno=" TARGET_ABI_FMT_ld " (%s)\n", -ret, target_strerror(-ret));

                } else {

                    gemu_log(" = " TARGET_ABI_FMT_ld "\n", ret);

                }

            }

            break;

        }

}
