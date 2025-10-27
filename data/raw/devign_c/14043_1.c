int index_from_key(const char *key)

{

    int i;



    for (i = 0; QKeyCode_lookup[i] != NULL; i++) {

        if (!strcmp(key, QKeyCode_lookup[i])) {

            break;

        }

    }



    /* Return Q_KEY_CODE__MAX if the key is invalid */

    return i;

}
