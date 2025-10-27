static int qemu_suspend_requested(void)

{

    int r = suspend_requested;

    suspend_requested = 0;

    return r;

}
