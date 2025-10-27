static int get_device_guid(

    char *name,

    int name_size,

    char *actual_name,

    int actual_name_size)

{

    LONG status;

    HKEY control_net_key;

    DWORD len;

    int i = 0;

    int stop = 0;



    status = RegOpenKeyEx(

        HKEY_LOCAL_MACHINE,

        NETWORK_CONNECTIONS_KEY,

        0,

        KEY_READ,

        &control_net_key);



    if (status != ERROR_SUCCESS) {

        return -1;

    }



    while (!stop)

    {

        char enum_name[256];

        char connection_string[256];

        HKEY connection_key;

        char name_data[256];

        DWORD name_type;

        const char name_string[] = "Name";



        len = sizeof (enum_name);

        status = RegEnumKeyEx(

            control_net_key,

            i,

            enum_name,

            &len,

            NULL,

            NULL,

            NULL,

            NULL);



        if (status == ERROR_NO_MORE_ITEMS)

            break;

        else if (status != ERROR_SUCCESS) {

            return -1;

        }



        snprintf(connection_string,

             sizeof(connection_string),

             "%s\\%s\\Connection",

             NETWORK_CONNECTIONS_KEY, enum_name);



        status = RegOpenKeyEx(

            HKEY_LOCAL_MACHINE,

            connection_string,

            0,

            KEY_READ,

            &connection_key);



        if (status == ERROR_SUCCESS) {

            len = sizeof (name_data);

            status = RegQueryValueEx(

                connection_key,

                name_string,

                NULL,

                &name_type,

                (LPBYTE)name_data,

                &len);



            if (status != ERROR_SUCCESS || name_type != REG_SZ) {

                    return -1;

            }

            else {

                if (is_tap_win32_dev(enum_name)) {

                    snprintf(name, name_size, "%s", enum_name);

                    if (actual_name) {

                        if (strcmp(actual_name, "") != 0) {

                            if (strcmp(name_data, actual_name) != 0) {

                                RegCloseKey (connection_key);

                                ++i;

                                continue;

                            }

                        }

                        else {

                            snprintf(actual_name, actual_name_size, "%s", name_data);

                        }

                    }

                    stop = 1;

                }

            }



            RegCloseKey (connection_key);

        }

        ++i;

    }



    RegCloseKey (control_net_key);



    if (stop == 0)

        return -1;



    return 0;

}
