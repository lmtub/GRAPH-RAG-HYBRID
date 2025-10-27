static const char *keyval_parse_one(QDict *qdict, const char *params,

                                    const char *implied_key,

                                    Error **errp)

{

    const char *key, *key_end, *s;

    size_t len;

    char key_in_cur[128];

    QDict *cur;

    QObject *next;

    QString *val;



    key = params;

    len = strcspn(params, "=,");

    if (implied_key && len && key[len] != '=') {

        /* Desugar implied key */

        key = implied_key;

        len = strlen(implied_key);

    }

    key_end = key + len;



    /*

     * Loop over key fragments: @s points to current fragment, it

     * applies to @cur.  @key_in_cur[] holds the previous fragment.

     */

    cur = qdict;

    s = key;

    for (;;) {

        for (len = 0; s + len < key_end && s[len] != '.'; len++) {

        }

        if (!len) {

            assert(key != implied_key);

            error_setg(errp, "Invalid parameter '%.*s'",

                       (int)(key_end - key), key);

            return NULL;

        }

        if (len >= sizeof(key_in_cur)) {

            assert(key != implied_key);

            error_setg(errp, "Parameter%s '%.*s' is too long",

                       s != key || s + len != key_end ? " fragment" : "",

                       (int)len, s);

            return NULL;

        }



        if (s != key) {

            next = keyval_parse_put(cur, key_in_cur, NULL,

                                    key, s - 1, errp);

            if (!next) {

                return NULL;

            }

            cur = qobject_to_qdict(next);

            assert(cur);

        }



        memcpy(key_in_cur, s, len);

        key_in_cur[len] = 0;

        s += len;



        if (*s != '.') {

            break;

        }

        s++;

    }



    if (key == implied_key) {

        assert(!*s);

        s = params;

    } else {

        if (*s != '=') {

            error_setg(errp, "Expected '=' after parameter '%.*s'",

                       (int)(s - key), key);

            return NULL;

        }

        s++;

    }



    val = qstring_new();

    for (;;) {

        if (!*s) {

            break;

        } else if (*s == ',') {

            s++;

            if (*s != ',') {

                break;

            }

        }

        qstring_append_chr(val, *s++);

    }



    if (!keyval_parse_put(cur, key_in_cur, val, key, key_end, errp)) {

        return NULL;

    }

    return s;

}
