static void test_ivshmem_memdev(void)

{

    IVState state;



    /* just for the sake of checking memory-backend property */

    setup_vm_cmd(&state, "-object memory-backend-ram,size=1M,id=mb1"

                 " -device ivshmem,x-memdev=mb1", false);



    qtest_quit(state.qtest);

}
