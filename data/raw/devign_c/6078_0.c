static int ppc_hash32_check_prot(int prot, int rw, int access_type)

{

    int ret;



    if (access_type == ACCESS_CODE) {

        if (prot & PAGE_EXEC) {

            ret = 0;

        } else {

            ret = -2;

        }

    } else if (rw) {

        if (prot & PAGE_WRITE) {

            ret = 0;

        } else {

            ret = -2;

        }

    } else {

        if (prot & PAGE_READ) {

            ret = 0;

        } else {

            ret = -2;

        }

    }



    return ret;

}
