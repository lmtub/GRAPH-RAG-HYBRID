static int64_t expr_unary(Monitor *mon)

{

    int64_t n;

    char *p;

    int ret;



    switch(*pch) {

    case '+':

        next();

        n = expr_unary(mon);

        break;

    case '-':

        next();

        n = -expr_unary(mon);

        break;

    case '~':

        next();

        n = ~expr_unary(mon);

        break;

    case '(':

        next();

        n = expr_sum(mon);

        if (*pch != ')') {

            expr_error(mon, "')' expected");

        }

        next();

        break;

    case '\'':

        pch++;

        if (*pch == '\0')

            expr_error(mon, "character constant expected");

        n = *pch;

        pch++;

        if (*pch != '\'')

            expr_error(mon, "missing terminating \' character");

        next();

        break;

    case '$':

        {

            char buf[128], *q;

            target_long reg=0;



            pch++;

            q = buf;

            while ((*pch >= 'a' && *pch <= 'z') ||

                   (*pch >= 'A' && *pch <= 'Z') ||

                   (*pch >= '0' && *pch <= '9') ||

                   *pch == '_' || *pch == '.') {

                if ((q - buf) < sizeof(buf) - 1)

                    *q++ = *pch;

                pch++;

            }

            while (qemu_isspace(*pch))

                pch++;

            *q = 0;

            ret = get_monitor_def(&reg, buf);

            if (ret == -1)

                expr_error(mon, "unknown register");

            else if (ret == -2)

                expr_error(mon, "no cpu defined");

            n = reg;

        }

        break;

    case '\0':

        expr_error(mon, "unexpected end of expression");

        n = 0;

        break;

    default:

#if TARGET_PHYS_ADDR_BITS > 32

        n = strtoull(pch, &p, 0);

#else

        n = strtoul(pch, &p, 0);

#endif

        if (pch == p) {

            expr_error(mon, "invalid char in expression");

        }

        pch = p;

        while (qemu_isspace(*pch))

            pch++;

        break;

    }

    return n;

}
