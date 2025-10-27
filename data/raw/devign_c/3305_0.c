static char *print_drive(void *ptr)

{

    return g_strdup(bdrv_get_device_name(ptr));

}
