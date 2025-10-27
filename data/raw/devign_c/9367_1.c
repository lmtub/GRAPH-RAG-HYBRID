static void multipath_pr_init(void)

{

    static struct udev *udev;



    udev = udev_new();

    mpath_lib_init(udev);

}
