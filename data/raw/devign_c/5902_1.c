static void utf8_string(void)

{

    /*

     * FIXME Current behavior for invalid UTF-8 sequences is

     * incorrect.  This test expects current, incorrect results.

     * They're all marked "bug:" below, and are to be replaced by

     * correct ones as the bugs get fixed.

     *

     * The JSON parser rejects some invalid sequences, but accepts

     * others without correcting the problem.

     *

     * We should either reject all invalid sequences, or minimize

     * overlong sequences and replace all other invalid sequences by a

     * suitable replacement character.  A common choice for

     * replacement is U+FFFD.

     *

     * Problem: we can't easily deal with embedded U+0000.  Parsing

     * the JSON string "this \\u0000" is fun" yields "this \0 is fun",

     * which gets misinterpreted as NUL-terminated "this ".  We should

     * consider using overlong encoding \xC0\x80 for U+0000 ("modified

     * UTF-8").

     *

     * Most test cases are scraped from Markus Kuhn's UTF-8 decoder

     * capability and stress test at

     * http://www.cl.cam.ac.uk/~mgk25/ucs/examples/UTF-8-test.txt

     */

    static const struct {

        const char *json_in;

        const char *utf8_out;

        const char *json_out;   /* defaults to @json_in */

        const char *utf8_in;    /* defaults to @utf8_out */

    } test_cases[] = {

        /*

         * Bug markers used here:

         * - bug: not corrected

         *   JSON parser fails to correct invalid sequence(s)

         * - bug: rejected

         *   JSON parser rejects invalid sequence(s)

         *   We may choose to define this as feature

         * - bug: want "..."

         *   JSON parser produces incorrect result, this is the

         *   correct one, assuming replacement character U+FFFF

         *   We may choose to reject instead of replace

         */



        /* 1  Some correct UTF-8 text */

        {

            /* a bit of German */

            "\"Falsches \xC3\x9C" "ben von Xylophonmusik qu\xC3\xA4lt"

            " jeden gr\xC3\xB6\xC3\x9F" "eren Zwerg.\"",

            "Falsches \xC3\x9C" "ben von Xylophonmusik qu\xC3\xA4lt"

            " jeden gr\xC3\xB6\xC3\x9F" "eren Zwerg.",

            "\"Falsches \\u00DCben von Xylophonmusik qu\\u00E4lt"

            " jeden gr\\u00F6\\u00DFeren Zwerg.\"",

        },

        {

            /* a bit of Greek */

            "\"\xCE\xBA\xE1\xBD\xB9\xCF\x83\xCE\xBC\xCE\xB5\"",

            "\xCE\xBA\xE1\xBD\xB9\xCF\x83\xCE\xBC\xCE\xB5",

            "\"\\u03BA\\u1F79\\u03C3\\u03BC\\u03B5\"",

        },

        /* 2  Boundary condition test cases */

        /* 2.1  First possible sequence of a certain length */

        /* 2.1.1  1 byte U+0000 */

        {

            "\"\\u0000\"",

            "",                 /* bug: want overlong "\xC0\x80" */

            "\"\\u0000\"",

            "\xC0\x80",

        },

        /* 2.1.2  2 bytes U+0080 */

        {

            "\"\xC2\x80\"",

            "\xC2\x80",

            "\"\\u0080\"",

        },

        /* 2.1.3  3 bytes U+0800 */

        {

            "\"\xE0\xA0\x80\"",

            "\xE0\xA0\x80",

            "\"\\u0800\"",

        },

        /* 2.1.4  4 bytes U+10000 */

        {

            "\"\xF0\x90\x80\x80\"",

            "\xF0\x90\x80\x80",

            "\"\\uD800\\uDC00\"",

        },

        /* 2.1.5  5 bytes U+200000 */

        {

            "\"\xF8\x88\x80\x80\x80\"",

            NULL,               /* bug: rejected */

            "\"\\uFFFD\"",

            "\xF8\x88\x80\x80\x80",

        },

        /* 2.1.6  6 bytes U+4000000 */

        {

            "\"\xFC\x84\x80\x80\x80\x80\"",

            NULL,               /* bug: rejected */

            "\"\\uFFFD\"",

            "\xFC\x84\x80\x80\x80\x80",

        },

        /* 2.2  Last possible sequence of a certain length */

        /* 2.2.1  1 byte U+007F */

        {

            "\"\x7F\"",

            "\x7F",

            "\"\\u007F\"",

        },

        /* 2.2.2  2 bytes U+07FF */

        {

            "\"\xDF\xBF\"",

            "\xDF\xBF",

            "\"\\u07FF\"",

        },

        /*

         * 2.2.3  3 bytes U+FFFC

         * The last possible sequence is actually U+FFFF.  But that's

         * a noncharacter, and already covered by its own test case

         * under 5.3.  Same for U+FFFE.  U+FFFD is the last character

         * in the BMP, and covered under 2.3.  Because of U+FFFD's

         * special role as replacement character, it's worth testing

         * U+FFFC here.

         */

        {

            "\"\xEF\xBF\xBC\"",

            "\xEF\xBF\xBC",

            "\"\\uFFFC\"",

        },

        /* 2.2.4  4 bytes U+1FFFFF */

        {

            "\"\xF7\xBF\xBF\xBF\"",

            NULL,               /* bug: rejected */

            "\"\\uFFFD\"",

            "\xF7\xBF\xBF\xBF",

        },

        /* 2.2.5  5 bytes U+3FFFFFF */

        {

            "\"\xFB\xBF\xBF\xBF\xBF\"",

            NULL,               /* bug: rejected */

            "\"\\uFFFD\"",

            "\xFB\xBF\xBF\xBF\xBF",

        },

        /* 2.2.6  6 bytes U+7FFFFFFF */

        {

            "\"\xFD\xBF\xBF\xBF\xBF\xBF\"",

            NULL,               /* bug: rejected */

            "\"\\uFFFD\"",

            "\xFD\xBF\xBF\xBF\xBF\xBF",

        },

        /* 2.3  Other boundary conditions */

        {

            /* last one before surrogate range: U+D7FF */

            "\"\xED\x9F\xBF\"",

            "\xED\x9F\xBF",

            "\"\\uD7FF\"",

        },

        {

            /* first one after surrogate range: U+E000 */

            "\"\xEE\x80\x80\"",

            "\xEE\x80\x80",

            "\"\\uE000\"",

        },

        {

            /* last one in BMP: U+FFFD */

            "\"\xEF\xBF\xBD\"",

            "\xEF\xBF\xBD",

            "\"\\uFFFD\"",

        },

        {

            /* last one in last plane: U+10FFFD */

            "\"\xF4\x8F\xBF\xBD\"",

            "\xF4\x8F\xBF\xBD",

            "\"\\uDBFF\\uDFFD\""

        },

        {

            /* first one beyond Unicode range: U+110000 */

            "\"\xF4\x90\x80\x80\"",

            "\xF4\x90\x80\x80",

            "\"\\uFFFD\"",

        },

        /* 3  Malformed sequences */

        /* 3.1  Unexpected continuation bytes */

        /* 3.1.1  First continuation byte */

        {

            "\"\x80\"",

            "\x80",             /* bug: not corrected */

            "\"\\uFFFD\"",

        },

        /* 3.1.2  Last continuation byte */

        {

            "\"\xBF\"",

            "\xBF",             /* bug: not corrected */

            "\"\\uFFFD\"",

        },

        /* 3.1.3  2 continuation bytes */

        {

            "\"\x80\xBF\"",

            "\x80\xBF",         /* bug: not corrected */

            "\"\\uFFFD\\uFFFD\"",

        },

        /* 3.1.4  3 continuation bytes */

        {

            "\"\x80\xBF\x80\"",

            "\x80\xBF\x80",     /* bug: not corrected */

            "\"\\uFFFD\\uFFFD\\uFFFD\"",

        },

        /* 3.1.5  4 continuation bytes */

        {

            "\"\x80\xBF\x80\xBF\"",

            "\x80\xBF\x80\xBF", /* bug: not corrected */

            "\"\\uFFFD\\uFFFD\\uFFFD\\uFFFD\"",

        },

        /* 3.1.6  5 continuation bytes */

        {

            "\"\x80\xBF\x80\xBF\x80\"",

            "\x80\xBF\x80\xBF\x80", /* bug: not corrected */

            "\"\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\"",

        },

        /* 3.1.7  6 continuation bytes */

        {

            "\"\x80\xBF\x80\xBF\x80\xBF\"",

            "\x80\xBF\x80\xBF\x80\xBF", /* bug: not corrected */

            "\"\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\"",

        },

        /* 3.1.8  7 continuation bytes */

        {

            "\"\x80\xBF\x80\xBF\x80\xBF\x80\"",

            "\x80\xBF\x80\xBF\x80\xBF\x80", /* bug: not corrected */

            "\"\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\"",

        },

        /* 3.1.9  Sequence of all 64 possible continuation bytes */

        {

            "\"\x80\x81\x82\x83\x84\x85\x86\x87"

            "\x88\x89\x8A\x8B\x8C\x8D\x8E\x8F"

            "\x90\x91\x92\x93\x94\x95\x96\x97"

            "\x98\x99\x9A\x9B\x9C\x9D\x9E\x9F"

            "\xA0\xA1\xA2\xA3\xA4\xA5\xA6\xA7"

            "\xA8\xA9\xAA\xAB\xAC\xAD\xAE\xAF"

            "\xB0\xB1\xB2\xB3\xB4\xB5\xB6\xB7"

            "\xB8\xB9\xBA\xBB\xBC\xBD\xBE\xBF\"",

             /* bug: not corrected */

            "\x80\x81\x82\x83\x84\x85\x86\x87"

            "\x88\x89\x8A\x8B\x8C\x8D\x8E\x8F"

            "\x90\x91\x92\x93\x94\x95\x96\x97"

            "\x98\x99\x9A\x9B\x9C\x9D\x9E\x9F"

            "\xA0\xA1\xA2\xA3\xA4\xA5\xA6\xA7"

            "\xA8\xA9\xAA\xAB\xAC\xAD\xAE\xAF"

            "\xB0\xB1\xB2\xB3\xB4\xB5\xB6\xB7"

            "\xB8\xB9\xBA\xBB\xBC\xBD\xBE\xBF",

            "\"\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD"

            "\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD"

            "\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD"

            "\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD"

            "\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD"

            "\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD"

            "\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD"

            "\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\""

        },

        /* 3.2  Lonely start characters */

        /* 3.2.1  All 32 first bytes of 2-byte sequences, followed by space */

        {

            "\"\xC0 \xC1 \xC2 \xC3 \xC4 \xC5 \xC6 \xC7 "

            "\xC8 \xC9 \xCA \xCB \xCC \xCD \xCE \xCF "

            "\xD0 \xD1 \xD2 \xD3 \xD4 \xD5 \xD6 \xD7 "

            "\xD8 \xD9 \xDA \xDB \xDC \xDD \xDE \xDF \"",

            NULL,               /* bug: rejected */

            "\"\\uFFFD \\uFFFD \\uFFFD \\uFFFD \\uFFFD \\uFFFD \\uFFFD \\uFFFD "

            "\\uFFFD \\uFFFD \\uFFFD \\uFFFD \\uFFFD \\uFFFD \\uFFFD \\uFFFD "

            "\\uFFFD \\uFFFD \\uFFFD \\uFFFD \\uFFFD \\uFFFD \\uFFFD \\uFFFD "

            "\\uFFFD \\uFFFD \\uFFFD \\uFFFD \\uFFFD \\uFFFD \\uFFFD \\uFFFD \"",

            "\xC0 \xC1 \xC2 \xC3 \xC4 \xC5 \xC6 \xC7 "

            "\xC8 \xC9 \xCA \xCB \xCC \xCD \xCE \xCF "

            "\xD0 \xD1 \xD2 \xD3 \xD4 \xD5 \xD6 \xD7 "

            "\xD8 \xD9 \xDA \xDB \xDC \xDD \xDE \xDF ",

        },

        /* 3.2.2  All 16 first bytes of 3-byte sequences, followed by space */

        {

            "\"\xE0 \xE1 \xE2 \xE3 \xE4 \xE5 \xE6 \xE7 "

            "\xE8 \xE9 \xEA \xEB \xEC \xED \xEE \xEF \"",

            /* bug: not corrected */

            "\xE0 \xE1 \xE2 \xE3 \xE4 \xE5 \xE6 \xE7 "

            "\xE8 \xE9 \xEA \xEB \xEC \xED \xEE \xEF ",

            "\"\\uFFFD \\uFFFD \\uFFFD \\uFFFD \\uFFFD \\uFFFD \\uFFFD \\uFFFD "

            "\\uFFFD \\uFFFD \\uFFFD \\uFFFD \\uFFFD \\uFFFD \\uFFFD \\uFFFD \"",

        },

        /* 3.2.3  All 8 first bytes of 4-byte sequences, followed by space */

        {

            "\"\xF0 \xF1 \xF2 \xF3 \xF4 \xF5 \xF6 \xF7 \"",

            NULL,               /* bug: rejected */

            "\"\\uFFFD \\uFFFD \\uFFFD \\uFFFD \\uFFFD \\uFFFD \\uFFFD \\uFFFD \"",

            "\xF0 \xF1 \xF2 \xF3 \xF4 \xF5 \xF6 \xF7 ",

        },

        /* 3.2.4  All 4 first bytes of 5-byte sequences, followed by space */

        {

            "\"\xF8 \xF9 \xFA \xFB \"",

            NULL,               /* bug: rejected */

            "\"\\uFFFD \\uFFFD \\uFFFD \\uFFFD \"",

            "\xF8 \xF9 \xFA \xFB ",

        },

        /* 3.2.5  All 2 first bytes of 6-byte sequences, followed by space */

        {

            "\"\xFC \xFD \"",

            NULL,               /* bug: rejected */

            "\"\\uFFFD \\uFFFD \"",

            "\xFC \xFD ",

        },

        /* 3.3  Sequences with last continuation byte missing */

        /* 3.3.1  2-byte sequence with last byte missing (U+0000) */

        {

            "\"\xC0\"",

            NULL,               /* bug: rejected */

            "\"\\uFFFD\"",

            "\xC0",

        },

        /* 3.3.2  3-byte sequence with last byte missing (U+0000) */

        {

            "\"\xE0\x80\"",

            "\xE0\x80",           /* bug: not corrected */

            "\"\\uFFFD\"",

        },

        /* 3.3.3  4-byte sequence with last byte missing (U+0000) */

        {

            "\"\xF0\x80\x80\"",

            "\xF0\x80\x80",     /* bug: not corrected */

            "\"\\uFFFD\"",

        },

        /* 3.3.4  5-byte sequence with last byte missing (U+0000) */

        {

            "\"\xF8\x80\x80\x80\"",

            NULL,                   /* bug: rejected */

            "\"\\uFFFD\"",

            "\xF8\x80\x80\x80",

        },

        /* 3.3.5  6-byte sequence with last byte missing (U+0000) */

        {

            "\"\xFC\x80\x80\x80\x80\"",

            NULL,                        /* bug: rejected */

            "\"\\uFFFD\"",

            "\xFC\x80\x80\x80\x80",

        },

        /* 3.3.6  2-byte sequence with last byte missing (U+07FF) */

        {

            "\"\xDF\"",

            "\xDF",             /* bug: not corrected */

            "\"\\uFFFD\"",

        },

        /* 3.3.7  3-byte sequence with last byte missing (U+FFFF) */

        {

            "\"\xEF\xBF\"",

            "\xEF\xBF",           /* bug: not corrected */

            "\"\\uFFFD\"",

        },

        /* 3.3.8  4-byte sequence with last byte missing (U+1FFFFF) */

        {

            "\"\xF7\xBF\xBF\"",

            NULL,               /* bug: rejected */

            "\"\\uFFFD\"",

            "\xF7\xBF\xBF",

        },

        /* 3.3.9  5-byte sequence with last byte missing (U+3FFFFFF) */

        {

            "\"\xFB\xBF\xBF\xBF\"",

            NULL,                 /* bug: rejected */

            "\"\\uFFFD\"",

            "\xFB\xBF\xBF\xBF",

        },

        /* 3.3.10  6-byte sequence with last byte missing (U+7FFFFFFF) */

        {

            "\"\xFD\xBF\xBF\xBF\xBF\"",

            NULL,                        /* bug: rejected */

            "\"\\uFFFD\"",

            "\xFD\xBF\xBF\xBF\xBF",

        },

        /* 3.4  Concatenation of incomplete sequences */

        {

            "\"\xC0\xE0\x80\xF0\x80\x80\xF8\x80\x80\x80\xFC\x80\x80\x80\x80"

            "\xDF\xEF\xBF\xF7\xBF\xBF\xFB\xBF\xBF\xBF\xFD\xBF\xBF\xBF\xBF\"",

            NULL,               /* bug: rejected */

            "\"\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD"

            "\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\"",

            "\xC0\xE0\x80\xF0\x80\x80\xF8\x80\x80\x80\xFC\x80\x80\x80\x80"

            "\xDF\xEF\xBF\xF7\xBF\xBF\xFB\xBF\xBF\xBF\xFD\xBF\xBF\xBF\xBF",

        },

        /* 3.5  Impossible bytes */

        {

            "\"\xFE\"",

            NULL,               /* bug: rejected */

            "\"\\uFFFD\"",

            "\xFE",

        },

        {

            "\"\xFF\"",

            NULL,               /* bug: rejected */

            "\"\\uFFFD\"",

            "\xFF",

        },

        {

            "\"\xFE\xFE\xFF\xFF\"",

            NULL,                 /* bug: rejected */

            "\"\\uFFFD\\uFFFD\\uFFFD\\uFFFD\"",

            "\xFE\xFE\xFF\xFF",

        },

        /* 4  Overlong sequences */

        /* 4.1  Overlong '/' */

        {

            "\"\xC0\xAF\"",

            NULL,               /* bug: rejected */

            "\"\\uFFFD\"",

            "\xC0\xAF",

        },

        {

            "\"\xE0\x80\xAF\"",

            "\xE0\x80\xAF",     /* bug: not corrected */

            "\"\\uFFFD\"",

        },

        {

            "\"\xF0\x80\x80\xAF\"",

            "\xF0\x80\x80\xAF",  /* bug: not corrected */

            "\"\\uFFFD\"",

        },

        {

            "\"\xF8\x80\x80\x80\xAF\"",

            NULL,                        /* bug: rejected */

            "\"\\uFFFD\"",

            "\xF8\x80\x80\x80\xAF",

        },

        {

            "\"\xFC\x80\x80\x80\x80\xAF\"",

            NULL,                               /* bug: rejected */

            "\"\\uFFFD\"",

            "\xFC\x80\x80\x80\x80\xAF",

        },

        /*

         * 4.2  Maximum overlong sequences

         * Highest Unicode value that is still resulting in an

         * overlong sequence if represented with the given number of

         * bytes.  This is a boundary test for safe UTF-8 decoders.

         */

        {

            /* \U+007F */

            "\"\xC1\xBF\"",

            NULL,               /* bug: rejected */

            "\"\\uFFFD\"",

            "\xC1\xBF",

        },

        {

            /* \U+07FF */

            "\"\xE0\x9F\xBF\"",

            "\xE0\x9F\xBF",     /* bug: not corrected */

            "\"\\uFFFD\"",

        },

        {

            /*

             * \U+FFFC

             * The actual maximum would be U+FFFF, but that's a

             * noncharacter.  Testing U+FFFC seems more useful.  See

             * also 2.2.3

             */

            "\"\xF0\x8F\xBF\xBC\"",

            "\xF0\x8F\xBF\xBC",   /* bug: not corrected */

            "\"\\uFFFD\"",

        },

        {

            /* \U+1FFFFF */

            "\"\xF8\x87\xBF\xBF\xBF\"",

            NULL,                        /* bug: rejected */

            "\"\\uFFFD\"",

            "\xF8\x87\xBF\xBF\xBF",

        },

        {

            /* \U+3FFFFFF */

            "\"\xFC\x83\xBF\xBF\xBF\xBF\"",

            NULL,                               /* bug: rejected */

            "\"\\uFFFD\"",

            "\xFC\x83\xBF\xBF\xBF\xBF",

        },

        /* 4.3  Overlong representation of the NUL character */

        {

            /* \U+0000 */

            "\"\xC0\x80\"",

            NULL,               /* bug: rejected */

            "\"\\u0000\"",

            "\xC0\x80",

        },

        {

            /* \U+0000 */

            "\"\xE0\x80\x80\"",

            "\xE0\x80\x80",     /* bug: not corrected */

            "\"\\uFFFD\"",

        },

        {

            /* \U+0000 */

            "\"\xF0\x80\x80\x80\"",

            "\xF0\x80\x80\x80",   /* bug: not corrected */

            "\"\\uFFFD\"",

        },

        {

            /* \U+0000 */

            "\"\xF8\x80\x80\x80\x80\"",

            NULL,                        /* bug: rejected */

            "\"\\uFFFD\"",

            "\xF8\x80\x80\x80\x80",

        },

        {

            /* \U+0000 */

            "\"\xFC\x80\x80\x80\x80\x80\"",

            NULL,                               /* bug: rejected */

            "\"\\uFFFD\"",

            "\xFC\x80\x80\x80\x80\x80",

        },

        /* 5  Illegal code positions */

        /* 5.1  Single UTF-16 surrogates */

        {

            /* \U+D800 */

            "\"\xED\xA0\x80\"",

            "\xED\xA0\x80",     /* bug: not corrected */

            "\"\\uFFFD\"",

        },

        {

            /* \U+DB7F */

            "\"\xED\xAD\xBF\"",

            "\xED\xAD\xBF",     /* bug: not corrected */

            "\"\\uFFFD\"",

        },

        {

            /* \U+DB80 */

            "\"\xED\xAE\x80\"",

            "\xED\xAE\x80",     /* bug: not corrected */

            "\"\\uFFFD\"",

        },

        {

            /* \U+DBFF */

            "\"\xED\xAF\xBF\"",

            "\xED\xAF\xBF",     /* bug: not corrected */

            "\"\\uFFFD\"",

        },

        {

            /* \U+DC00 */

            "\"\xED\xB0\x80\"",

            "\xED\xB0\x80",     /* bug: not corrected */

            "\"\\uFFFD\"",

        },

        {

            /* \U+DF80 */

            "\"\xED\xBE\x80\"",

            "\xED\xBE\x80",     /* bug: not corrected */

            "\"\\uFFFD\"",

        },

        {

            /* \U+DFFF */

            "\"\xED\xBF\xBF\"",

            "\xED\xBF\xBF",     /* bug: not corrected */

            "\"\\uFFFD\"",

        },

        /* 5.2  Paired UTF-16 surrogates */

        {

            /* \U+D800\U+DC00 */

            "\"\xED\xA0\x80\xED\xB0\x80\"",

            "\xED\xA0\x80\xED\xB0\x80", /* bug: not corrected */

            "\"\\uFFFD\\uFFFD\"",

        },

        {

            /* \U+D800\U+DFFF */

            "\"\xED\xA0\x80\xED\xBF\xBF\"",

            "\xED\xA0\x80\xED\xBF\xBF", /* bug: not corrected */

            "\"\\uFFFD\\uFFFD\"",

        },

        {

            /* \U+DB7F\U+DC00 */

            "\"\xED\xAD\xBF\xED\xB0\x80\"",

            "\xED\xAD\xBF\xED\xB0\x80", /* bug: not corrected */

            "\"\\uFFFD\\uFFFD\"",

        },

        {

            /* \U+DB7F\U+DFFF */

            "\"\xED\xAD\xBF\xED\xBF\xBF\"",

            "\xED\xAD\xBF\xED\xBF\xBF", /* bug: not corrected */

            "\"\\uFFFD\\uFFFD\"",

        },

        {

            /* \U+DB80\U+DC00 */

            "\"\xED\xAE\x80\xED\xB0\x80\"",

            "\xED\xAE\x80\xED\xB0\x80", /* bug: not corrected */

            "\"\\uFFFD\\uFFFD\"",

        },

        {

            /* \U+DB80\U+DFFF */

            "\"\xED\xAE\x80\xED\xBF\xBF\"",

            "\xED\xAE\x80\xED\xBF\xBF", /* bug: not corrected */

            "\"\\uFFFD\\uFFFD\"",

        },

        {

            /* \U+DBFF\U+DC00 */

            "\"\xED\xAF\xBF\xED\xB0\x80\"",

            "\xED\xAF\xBF\xED\xB0\x80", /* bug: not corrected */

            "\"\\uFFFD\\uFFFD\"",

        },

        {

            /* \U+DBFF\U+DFFF */

            "\"\xED\xAF\xBF\xED\xBF\xBF\"",

            "\xED\xAF\xBF\xED\xBF\xBF", /* bug: not corrected */

            "\"\\uFFFD\\uFFFD\"",

        },

        /* 5.3  Other illegal code positions */

        /* BMP noncharacters */

        {

            /* \U+FFFE */

            "\"\xEF\xBF\xBE\"",

            "\xEF\xBF\xBE",     /* bug: not corrected */

            "\"\\uFFFD\"",

        },

        {

            /* \U+FFFF */

            "\"\xEF\xBF\xBF\"",

            "\xEF\xBF\xBF",     /* bug: not corrected */

            "\"\\uFFFD\"",

        },

        {

            /* U+FDD0 */

            "\"\xEF\xB7\x90\"",

            "\xEF\xB7\x90",     /* bug: not corrected */

            "\"\\uFFFD\"",

        },

        {

            /* U+FDEF */

            "\"\xEF\xB7\xAF\"",

            "\xEF\xB7\xAF",     /* bug: not corrected */

            "\"\\uFFFD\"",

        },

        /* Plane 1 .. 16 noncharacters */

        {

            /* U+1FFFE U+1FFFF U+2FFFE U+2FFFF ... U+10FFFE U+10FFFF */

            "\"\xF0\x9F\xBF\xBE\xF0\x9F\xBF\xBF"

            "\xF0\xAF\xBF\xBE\xF0\xAF\xBF\xBF"

            "\xF0\xBF\xBF\xBE\xF0\xBF\xBF\xBF"

            "\xF1\x8F\xBF\xBE\xF1\x8F\xBF\xBF"

            "\xF1\x9F\xBF\xBE\xF1\x9F\xBF\xBF"

            "\xF1\xAF\xBF\xBE\xF1\xAF\xBF\xBF"

            "\xF1\xBF\xBF\xBE\xF1\xBF\xBF\xBF"

            "\xF2\x8F\xBF\xBE\xF2\x8F\xBF\xBF"

            "\xF2\x9F\xBF\xBE\xF2\x9F\xBF\xBF"

            "\xF2\xAF\xBF\xBE\xF2\xAF\xBF\xBF"

            "\xF2\xBF\xBF\xBE\xF2\xBF\xBF\xBF"

            "\xF3\x8F\xBF\xBE\xF3\x8F\xBF\xBF"

            "\xF3\x9F\xBF\xBE\xF3\x9F\xBF\xBF"

            "\xF3\xAF\xBF\xBE\xF3\xAF\xBF\xBF"

            "\xF3\xBF\xBF\xBE\xF3\xBF\xBF\xBF"

            "\xF4\x8F\xBF\xBE\xF4\x8F\xBF\xBF\"",

            /* bug: not corrected */

            "\xF0\x9F\xBF\xBE\xF0\x9F\xBF\xBF"

            "\xF0\xAF\xBF\xBE\xF0\xAF\xBF\xBF"

            "\xF0\xBF\xBF\xBE\xF0\xBF\xBF\xBF"

            "\xF1\x8F\xBF\xBE\xF1\x8F\xBF\xBF"

            "\xF1\x9F\xBF\xBE\xF1\x9F\xBF\xBF"

            "\xF1\xAF\xBF\xBE\xF1\xAF\xBF\xBF"

            "\xF1\xBF\xBF\xBE\xF1\xBF\xBF\xBF"

            "\xF2\x8F\xBF\xBE\xF2\x8F\xBF\xBF"

            "\xF2\x9F\xBF\xBE\xF2\x9F\xBF\xBF"

            "\xF2\xAF\xBF\xBE\xF2\xAF\xBF\xBF"

            "\xF2\xBF\xBF\xBE\xF2\xBF\xBF\xBF"

            "\xF3\x8F\xBF\xBE\xF3\x8F\xBF\xBF"

            "\xF3\x9F\xBF\xBE\xF3\x9F\xBF\xBF"

            "\xF3\xAF\xBF\xBE\xF3\xAF\xBF\xBF"

            "\xF3\xBF\xBF\xBE\xF3\xBF\xBF\xBF"

            "\xF4\x8F\xBF\xBE\xF4\x8F\xBF\xBF",

            "\"\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD"

            "\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD"

            "\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD"

            "\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\\uFFFD\"",

        },

        {}

    };

    int i;

    QObject *obj;

    QString *str;

    const char *json_in, *utf8_out, *utf8_in, *json_out;



    for (i = 0; test_cases[i].json_in; i++) {

        json_in = test_cases[i].json_in;

        utf8_out = test_cases[i].utf8_out;

        utf8_in = test_cases[i].utf8_in ?: test_cases[i].utf8_out;

        json_out = test_cases[i].json_out ?: test_cases[i].json_in;



        obj = qobject_from_json(json_in, NULL);

        if (utf8_out) {

            str = qobject_to_qstring(obj);

            g_assert(str);

            g_assert_cmpstr(qstring_get_str(str), ==, utf8_out);

        } else {

            g_assert(!obj);

        }

        qobject_decref(obj);



        obj = QOBJECT(qstring_from_str(utf8_in));

        str = qobject_to_json(obj);

        if (json_out) {

            g_assert(str);

            g_assert_cmpstr(qstring_get_str(str), ==, json_out);

        } else {

            g_assert(!str);

        }

        QDECREF(str);

        qobject_decref(obj);



        /*

         * Disabled, because qobject_from_json() is buggy, and I can't

         * be bothered to add the expected incorrect results.

         * FIXME Enable once these bugs have been fixed.

         */

        if (0 && json_out != json_in) {

            obj = qobject_from_json(json_out, NULL);

            str = qobject_to_qstring(obj);

            g_assert(str);

            g_assert_cmpstr(qstring_get_str(str), ==, utf8_out);

        }

    }

}
