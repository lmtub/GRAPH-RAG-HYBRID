static QString *qstring_from_escaped_str(JSONParserContext *ctxt, QObject *token)

{

    const char *ptr = token_get_value(token);

    QString *str;

    int double_quote = 1;



    if (*ptr == '"') {

        double_quote = 1;

    } else {

        double_quote = 0;

    }

    ptr++;



    str = qstring_new();

    while (*ptr && 

           ((double_quote && *ptr != '"') || (!double_quote && *ptr != '\''))) {

        if (*ptr == '\\') {

            ptr++;



            switch (*ptr) {

            case '"':

                qstring_append(str, "\"");

                ptr++;

                break;

            case '\'':

                qstring_append(str, "'");

                ptr++;

                break;

            case '\\':

                qstring_append(str, "\\");

                ptr++;

                break;

            case '/':

                qstring_append(str, "/");

                ptr++;

                break;

            case 'b':

                qstring_append(str, "\b");

                ptr++;

                break;

            case 'f':

                qstring_append(str, "\f");

                ptr++;

                break;

            case 'n':

                qstring_append(str, "\n");

                ptr++;

                break;

            case 'r':

                qstring_append(str, "\r");

                ptr++;

                break;

            case 't':

                qstring_append(str, "\t");

                ptr++;

                break;

            case 'u': {

                uint16_t unicode_char = 0;

                char utf8_char[4];

                int i = 0;



                ptr++;



                for (i = 0; i < 4; i++) {

                    if (qemu_isxdigit(*ptr)) {

                        unicode_char |= hex2decimal(*ptr) << ((3 - i) * 4);

                    } else {

                        parse_error(ctxt, token,

                                    "invalid hex escape sequence in string");

                        goto out;

                    }

                    ptr++;

                }



                wchar_to_utf8(unicode_char, utf8_char, sizeof(utf8_char));

                qstring_append(str, utf8_char);

            }   break;

            default:

                parse_error(ctxt, token, "invalid escape sequence in string");

                goto out;

            }

        } else {

            char dummy[2];



            dummy[0] = *ptr++;

            dummy[1] = 0;



            qstring_append(str, dummy);

        }

    }



    return str;



out:

    QDECREF(str);

    return NULL;

}
