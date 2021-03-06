import json
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from dataprocessing import parse_json, business_path, review_path, user_path
from sklearn.manifold import TSNE
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
from surprise import SVD
import plotly.express as px
import datapane as dp
from collections import Counter


def reviews_per_user():
        # userDF = parse_json(user_path)


    # print(userDF.columns)

    # reviewCountList = userDF['review_count'].tolist()
    # reviewCountDic = {}
    # for value in reviewCountList:
    #     if value not in reviewCountDic:
    #         reviewCountDic[value] = 1
    #     else:
    #         reviewCountDic[value] += 1


    reviewCountDic = {1220: 4, 2136: 1, 119: 732, 987: 7, 495: 40, 229: 208, 51: 3625, 299: 90, 288: 113, 44: 4736, 65: 2341, 750: 8, 632: 15, 2: 271248, 363: 64, 356: 85, 77: 1662, 202: 270, 53: 3391, 1981: 1, 267: 143, 147: 488, 311: 101, 82: 1497, 657: 19, 12: 39974, 775: 10, 359: 68, 19: 20541, 252: 142, 127: 700, 80: 1595, 683: 10, 1370: 1, 40: 5520, 631: 20, 279: 130, 796: 6, 388: 54, 662: 22, 115: 812, 114: 858, 540: 18, 707: 18, 307: 108, 91: 1198, 160: 449, 61: 2688, 2689: 1, 598: 14, 376: 56, 872: 4, 4: 156239, 1503: 4, 74: 1861, 110: 829, 58: 2825, 4205: 1, 260: 155, 374: 48, 389: 56, 2392: 1, 671: 16, 90: 1237, 57: 2940, 228: 192, 46: 4411, 164: 404, 32: 8556, 7: 79590, 26: 12298, 111: 766, 195: 287, 60: 2613, 14: 32597, 478: 29, 100: 1027, 54: 3256, 41: 5408, 50: 3690, 96: 1096, 1907: 1, 24: 13938, 955: 5, 47: 4224, 141: 554, 193: 280, 555: 24, 129: 653, 143: 556, 33: 8063, 123: 697, 27: 11344, 215: 239, 682: 10, 3192: 1, 761: 11, 2149: 1, 419: 45, 798: 12, 117: 762, 176: 337, 148: 493, 475: 29, 1520: 1, 154: 465, 18: 22047, 84: 1408, 16: 26744, 22: 16189, 308: 108, 205: 248, 28: 10718, 30: 9550, 72: 1970, 1863: 3, 544: 22, 221: 231, 68: 2015, 104: 929, 467: 42, 372: 76, 1283: 1, 21: 17296, 63: 2374, 207: 288, 399: 53, 762: 14, 249: 180, 504: 32, 36: 6809, 31: 8985, 658: 16, 233: 205, 9: 58199, 1716: 2, 471: 41, 10: 51336, 2181: 1, 524: 20, 301: 116, 401: 62, 107: 924, 101: 1051, 139: 610, 106: 910, 109: 858, 1949: 1, 126: 660, 357: 74, 619: 11, 617: 21, 3531: 2, 240: 183, 131: 588, 1169: 5, 1104: 5, 23: 14765, 78: 1654, 817: 5, 962: 3, 25: 12953, 2469: 1, 268: 141, 238: 200, 120: 731, 287: 124, 99: 948, 38: 6259, 241: 162, 998: 3, 97: 1043, 13: 35882, 87: 1309, 103: 982, 45: 4549, 69: 2133, 531: 28, 122: 719, 5: 120966, 840: 11, 201: 269, 1365: 3, 43: 4924, 1512: 2, 1989: 2, 411: 59, 1685: 3, 795: 7, 302: 98, 209: 261, 2049: 1, 454: 44, 246: 172, 329: 81, 145: 509, 197: 279, 37: 6494, 251: 171, 1911: 1, 328: 98, 487: 40, 88: 1300, 1443: 1, 265: 158, 124: 685, 1084: 4, 243: 174, 579: 24, 92: 1204, 1126: 4, 322: 98, 814: 9, 1871: 2, 208: 228, 258: 155, 4308: 1, 262: 135, 2750: 1, 283: 123, 9906: 1, 759: 11, 1189: 3, 571: 25, 695: 15, 2042: 1, 655: 20, 922: 6, 35: 7309, 151: 473, 646: 22, 438: 46, 162: 387, 305: 106, 289: 107, 1930: 2, 6: 96853, 48: 4023, 75: 1748, 59: 2821, 29: 10111, 222: 229, 522: 32, 943: 7, 2455: 1, 185: 303, 118: 735, 652: 21, 213: 212, 198: 229, 170: 383, 343: 75, 158: 448, 2310: 1, 245: 185, 565: 30, 73: 1808, 586: 19, 2845: 1, 500: 34, 684: 12, 79: 1591, 300: 92, 15: 29344, 86: 1366, 168: 358, 42: 5139, 752: 7, 340: 97, 595: 29, 1754: 1, 1515: 4, 224: 223, 1991: 1, 318: 92, 116: 786, 337: 80, 280: 129, 1266: 3, 694: 7, 183: 319, 1568: 6, 1546: 5, 1290: 4, 851: 8, 932: 9, 412: 58, 2438: 1, 620: 14, 876: 7, 1261: 2, 535: 26, 2220: 1, 699: 11, 409: 45, 194: 309, 1705: 3, 587: 13, 396: 54, 270: 139, 3: 208749, 190: 278, 720: 10, 2602: 1, 486: 42, 81: 1546, 841: 5, 1182: 3, 406: 67, 244: 167, 232: 188, 98: 986, 8: 67241, 2751: 1, 1532: 2, 173: 361, 157: 466, 1003: 2, 339: 91, 71: 1990, 56: 3150, 461: 33, 331: 101, 346: 89, 1401: 5, 466: 39, 20: 18725, 502: 36, 11: 44481, 1072: 4, 1053: 10, 344: 78, 153: 480, 327: 92, 377: 68, 182: 308, 383: 55, 378: 70, 573: 27, 604: 21, 261: 157, 480: 33, 146: 541, 1476: 2, 149: 519, 323: 111, 102: 1032, 366: 68, 1987: 3, 161: 426, 527: 28, 1077: 4, 574: 25, 334: 91, 93: 1109, 360: 84, 614: 31, 278: 130, 1259: 2, 402: 58, 39: 5926, 184: 318, 663: 19, 562: 25, 216: 234, 1198: 3, 600: 22, 121: 708, 112: 868, 76: 1789, 465: 44, 397: 48, 192: 289, 1738: 4, 135: 579, 965: 8, 885: 6, 390: 70, 3168: 1, 341: 94, 49: 3819, 70: 1953, 425: 54, 163: 426, 1538: 1, 326: 94, 520: 27, 428: 57, 247: 160, 155: 425, 594: 18, 286: 114, 3175: 1, 186: 317, 370: 73, 113: 822, 470: 49, 292: 105, 642: 22, 995: 5, 85: 1309, 134: 583, 416: 59, 17: 24174, 2476: 1, 34: 7447, 590: 20, 255: 161, 177: 342, 1209: 3, 142: 503, 263: 175, 203: 269, 1509: 1, 1441: 2, 1413: 2, 362: 69, 749: 11, 276: 122, 335: 100, 1872: 1, 392: 65, 136: 617, 226: 216, 860: 5, 1065: 5, 1314: 2, 1986: 2, 490: 34, 391: 57, 1133: 4, 140: 547, 64: 2404, 1406: 3, 945: 6, 457: 25, 55: 3177, 150: 455, 426: 37, 62: 2536, 440: 53, 187: 292, 234: 185, 128: 604, 225: 192, 781: 13, 169: 390, 593: 25, 714: 18, 314: 89, 746: 13, 319: 101, 220: 236, 1009: 5, 1670: 2, 235: 174, 320: 110, 901: 8, 1161: 4, 293: 109, 189: 286, 237: 169, 1047: 5, 576: 29, 108: 868, 731: 7, 1691: 1, 1195: 1, 199: 317, 290: 112, 986: 4, 166: 367, 89: 1255, 907: 4, 5280: 1, 369: 65, 767: 13, 178: 357, 726: 12, 989: 5, 325: 90, 634: 23, 210: 270, 3978: 1, 436: 54, 167: 399, 66: 2155, 204: 280, 703: 13, 191: 281, 1036: 4, 622: 26, 1014: 5, 83: 1430, 2377: 2, 1608: 1, 882: 11, 171: 349, 144: 494, 405: 50, 747: 11, 439: 46, 831: 11, 538: 16, 400: 63, 629: 19, 445: 42, 2088: 1, 298: 115, 673: 21, 1218: 5, 645: 24, 214: 237, 696: 16, 206: 264, 2218: 1, 133: 643, 1023: 8, 333: 61, 188: 283, 602: 20, 52: 3488, 394: 41, 420: 50, 568: 24, 273: 141, 408: 62, 281: 127, 1204: 5, 271: 133, 1815: 1, 787: 15, 585: 20, 2293: 1, 491: 22, 441: 46, 338: 90, 306: 115, 628: 15, 67: 2197, 591: 23, 675: 14, 2138: 1, 1207: 3, 223: 219, 528: 31, 832: 2, 961: 2, 1080: 5, 342: 94, 165: 355, 217: 197, 559: 22, 1709: 2, 1170: 1, 2035: 2, 542: 26, 257: 176, 1018: 3, 1664: 3, 1090: 5, 584: 21, 515: 20, 137: 534, 545: 34, 916: 5, 890: 3, 94: 1104, 427: 48, 219: 207, 865: 6, 615: 14, 1223: 2, 743: 16, 403: 50, 413: 46, 3215: 1, 1384: 6, 132: 566, 1506: 1, 264: 149, 175: 342, 823: 11, 776: 15, 1311: 3, 691: 11, 172: 352, 462: 36, 174: 372, 285: 114, 442: 41, 304: 105, 354: 74, 754: 9, 596: 24, 1493: 4, 685: 15, 507: 33, 608: 24, 802: 10, 1783: 2, 606: 27, 248: 181, 95: 1088, 834: 7, 1094: 3, 873: 6, 138: 577, 242: 182, 1118: 4, 1096: 6, 560: 21, 477: 37, 517: 24, 1485: 2, 626: 16, 510: 24, 489: 28, 700: 10, 1354: 3, 463: 33, 780: 15, 836: 15, 1100: 4, 532: 26, 1148: 1, 1320: 5, 358: 70, 958: 3, 212: 231, 2870: 1, 488: 29, 12865: 1, 514: 28, 464: 40, 1356: 2, 348: 77, 239: 194, 1251: 4, 3162: 1, 382: 65, 951: 5, 854: 9, 874: 12, 1164: 2, 1287: 2, 549: 29, 1280: 3, 1019: 8, 458: 35, 236: 189, 1197: 3, 637: 20, 755: 12, 1273: 3, 1454: 2, 105: 910, 159: 428, 516: 34, 1486: 4, 393: 60, 485: 37, 250: 164, 180: 322, 230: 201, 421: 52, 640: 14, 688: 14, 1264: 3, 398: 51, 1294: 3, 315: 107, 218: 237, 2256: 1, 572: 25, 739: 15, 1291: 2, 492: 31, 1002: 5, 1627: 2, 534: 29, 494: 29, 563: 22, 125: 619, 953: 5, 179: 314, 661: 11, 548: 29, 1: 352766, 518: 42, 763: 8, 1059: 3, 564: 31, 920: 8, 1030: 4, 2013: 1, 503: 30, 367: 78, 1376: 1, 678: 10, 801: 9, 1792: 1, 130: 612, 437: 48, 680: 14, 949: 4, 1141: 3, 2154: 2, 296: 94, 472: 46, 1833: 2, 737: 11, 379: 74, 211: 234, 1825: 2, 1660: 1, 259: 171, 415: 61, 954: 5, 1095: 4, 355: 81, 350: 86, 275: 111, 610: 16, 295: 111, 417: 61, 321: 90, 291: 106, 381: 56, 972: 5, 758: 9, 444: 40, 1107: 4, 181: 339, 227: 217, 567: 30, 368: 56, 804: 6, 294: 124, 884: 8, 3844: 1, 269: 120, 303: 112, 455: 40, 698: 14, 1171: 9, 774: 8, 790: 7, 659: 22, 701: 7, 7741: 1, 1146: 4, 533: 29, 493: 37, 2356: 1, 635: 23, 904: 5, 1262: 5, 993: 3, 156: 426, 597: 24, 513: 24, 624: 17, 386: 52, 1411: 1, 200: 248, 317: 105, 1060: 7, 468: 36, 611: 15, 266: 151, 2440: 1, 448: 37, 274: 130, 525: 32, 536: 26, 1674: 3, 1555: 3, 638: 15, 523: 26, 794: 9, 566: 20, 999: 9, 254: 176, 1380: 2, 1037: 5, 665: 13, 793: 12, 481: 38, 347: 78, 788: 10, 435: 47, 704: 16, 349: 78, 850: 13, 1677: 1, 297: 97, 152: 442, 1081: 4, 1696: 2, 713: 8, 1071: 5, 613: 18, 588: 23, 316: 76, 592: 16, 456: 47, 1358: 2, 2548: 1, 782: 8, 745: 9, 1055: 5, 430: 54, 385: 61, 666: 16, 1255: 4, 1112: 3, 693: 9, 541: 24, 643: 16, 418: 48, 4188: 1, 539: 15, 423: 48, 810: 10, 1558: 1, 371: 61, 3076: 1, 936: 10, 484: 34, 1744: 3, 1388: 5, 537: 32, 897: 3, 1367: 3, 253: 161, 785: 8, 654: 19, 2241: 2, 434: 35, 770: 13, 603: 16, 735: 15, 4884: 2, 353: 84, 2195: 1, 1106: 3, 809: 9, 1763: 1, 373: 75, 282: 102, 918: 6, 1131: 4, 324: 80, 578: 19, 551: 24, 1695: 2, 1151: 8, 1061: 4, 2862: 1, 633: 14, 653: 18, 1039: 6, 899: 7, 963: 4, 1017: 3, 5415: 1, 512: 31, 432: 41, 1070: 3, 723: 13, 2906: 1, 361: 68, 284: 120, 1026: 10, 1387: 3, 1328: 2, 1452: 1, 1434: 2, 725: 10, 892: 8, 4207: 1, 1474: 1, 499: 26, 1010: 3, 352: 82, 609: 19, 1286: 2, 1025: 5, 4118: 1, 783: 9, 710: 9, 881: 10, 1421: 2, 1176: 2, 829: 3, 946: 7, 1656: 3, 575: 28, 256: 156, 828: 9, 1820: 2, 742: 16, 664: 16, 476: 30, 8066: 1, 509: 26, 1048: 3, 452: 50, 1955: 1, 1735: 1, 1473: 1, 724: 12, 2724: 2, 469: 36, 1420: 2, 927: 10, 450: 51, 6764: 1, 1462: 1, 506: 29, 404: 54, 616: 7, 2298: 1, 1642: 3, 1302: 2, 443: 43, 2203: 1, 530: 24, 623: 20, 2171: 3, 1335: 1, 976: 2, 424: 52, 447: 37, 5854: 1, 1632: 1, 351: 94, 717: 13, 1672: 1, 431: 53, 2514: 1, 1289: 1, 2682: 1, 846: 4, 891: 9, 429: 55, 556: 21, 1281: 4, 582: 22, 1625: 3, 4920: 1, 482: 28, 312: 85, 1542: 2, 330: 90, 667: 12, 1553: 1, 272: 138, 692: 12, 1654: 1, 497: 37, 607: 21, 1079: 2, 1562: 3, 2424: 1, 1728: 3, 459: 38, 496: 39, 1935: 2, 1858: 1, 2270: 1, 395: 63, 869: 8, 2120: 2, 558: 23, 2847: 1, 864: 6, 843: 10, 821: 7, 577: 30, 1966: 1, 689: 9, 6657: 1, 2599: 1, 896: 6, 668: 14, 601: 20, 784: 6, 380: 73, 1447: 1, 705: 14, 1824: 1, 599: 12, 1172: 3, 526: 29, 2946: 1, 1329: 2, 627: 10, 4880: 1, 3247: 1, 364: 69, 728: 8, 501: 30, 1737: 1, 1231: 3, 1786: 2, 964: 5, 1322: 2, 310: 114, 4030: 1, 825: 10, 808: 5, 1428: 3, 505: 23, 816: 9, 1415: 2, 656: 15, 1028: 3, 1533: 5, 1056: 3, 649: 20, 651: 18, 543: 17, 3155: 1, 740: 12, 621: 19, 1507: 2, 2143: 3, 410: 53, 820: 13, 887: 8, 910: 5, 674: 17, 789: 10, 2586: 1, 511: 29, 2117: 1, 414: 50, 641: 13, 1174: 5, 1752: 3, 947: 6, 460: 42, 2028: 1, 1063: 5, 1316: 2, 1013: 5, 196: 291, 553: 26, 1214: 3, 375: 69, 453: 44, 877: 5, 561: 29, 727: 12, 365: 65, 1524: 2, 422: 45, 2368: 1, 1364: 2, 1727: 3, 676: 25, 2666: 1, 712: 17, 1032: 3, 679: 19, 2217: 1, 1598: 1, 2722: 2, 985: 5, 1150: 3, 1334: 1, 826: 13, 3187: 1, 779: 11, 1521: 1, 1516: 2, 1860: 1, 1340: 3, 853: 10, 909: 5, 681: 15, 508: 31, 1181: 3, 1230: 1, 570: 28, 644: 17, 630: 19, 277: 118, 933: 6, 1988: 3, 1540: 4, 1551: 1, 1101: 5, 547: 31, 1536: 1, 894: 7, 4847: 1, 1038: 2, 309: 85, 2133: 1, 451: 36, 2100: 1, 830: 9, 760: 10, 3933: 1, 479: 30, 792: 7, 690: 11, 960: 5, 697: 9, 3119: 1, 978: 3, 1797: 3, 474: 39, 1122: 4, 2981: 1, 1741: 1, 1208: 2, 803: 7, 902: 9, 1117: 9, 824: 9, 729: 8, 1932: 1, 648: 16, 797: 7, 612: 22, 1199: 3, 384: 49, 1767: 2, 744: 10, 2010: 1, 2466: 1, 1903: 2, 786: 10, 581: 27, 1187: 4, 1069: 6, 855: 7, 1099: 3, 2836: 1, 1192: 2, 941: 8, 768: 8, 2651: 1, 636: 16, 979: 2, 686: 8, 1119: 6, 625: 15, 639: 15, 1179: 3, 977: 5, 550: 22, 2561: 1, 815: 8, 1215: 3, 1303: 7, 1588: 1, 1941: 1, 2084: 1, 811: 11, 1636: 1, 886: 1, 948: 10, 1446: 2, 1082: 8, 1484: 2, 2126: 1, 1066: 5, 827: 8, 387: 53, 914: 3, 719: 10, 557: 21, 313: 104, 2658: 1, 845: 8, 1301: 3, 1830: 1, 1564: 1, 3047: 1, 521: 39, 822: 6, 529: 20, 1031: 5, 2098: 1, 1000: 8, 818: 7, 546: 29, 1951: 1, 1092: 2, 718: 11, 1594: 1, 1414: 1, 14601: 1, 1128: 2, 1163: 3, 332: 107, 552: 28, 589: 21, 2360: 1, 1425: 1, 449: 38, 1481: 4, 1534: 4, 2346: 1, 1782: 1, 751: 11, 1772: 2, 906: 5, 1293: 6, 1075: 4, 1351: 3, 1130: 4, 483: 31, 1396: 4, 345: 65, 1369: 3, 583: 24, 883: 8, 1232: 5, 748: 11, 2071: 1, 498: 20, 1298: 3, 1777: 1, 1202: 1, 2911: 1, 1513: 1, 1258: 1, 771: 14, 866: 8, 1184: 4, 1855: 1, 1431: 2, 1730: 2, 5591: 1, 4000: 1, 2247: 1, 2359: 1, 974: 4, 756: 6, 672: 16, 231: 195, 1086: 4, 1078: 5, 1822: 1, 940: 3, 1020: 3, 1108: 3, 730: 12, 702: 12, 1011: 7, 1345: 4, 1412: 1, 1029: 5, 715: 13, 1186: 1, 769: 11, 3477: 1, 863: 8, 1445: 1, 1668: 3, 1416: 2, 407: 44, 1498: 1, 1488: 2, 764: 10, 1720: 1, 819: 10, 1836: 1, 1373: 2, 1158: 4, 980: 7, 900: 9, 1560: 2, 1992: 1, 879: 4, 1217: 1, 935: 2, 660: 14, 708: 11, 4004: 1, 984: 5, 1575: 2, 888: 3, 2572: 1, 1346: 1, 2531: 2, 580: 12, 844: 10, 1393: 2, 722: 11, 772: 9, 1058: 4, 446: 44, 433: 31, 1006: 8, 1052: 5, 734: 10, 911: 6, 1049: 6, 670: 15, 1295: 2, 1587: 2, 867: 7, 919: 6, 1085: 8, 1109: 4, 336: 95, 915: 4, 926: 3, 1734: 2, 6400: 1, 1267: 3, 925: 5, 903: 8, 753: 10, 1699: 1, 1494: 2, 2034: 1, 1114: 5, 859: 7, 1105: 5, 1153: 2, 2665: 1, 1088: 2, 519: 23, 1563: 2, 1134: 3, 778: 6, 3053: 1, 1042: 5, 1319: 2, 1203: 3, 736: 9, 1225: 4, 618: 14, 878: 6, 1487: 1, 799: 12, 1102: 2, 847: 10, 992: 3, 2199: 1, 1448: 3, 1761: 2, 2127: 1, 2533: 1, 1277: 2, 805: 6, 1407: 4, 2945: 1, 716: 6, 1111: 3, 669: 9, 1243: 2, 898: 2, 1016: 4, 981: 1, 889: 4, 1041: 7, 1548: 1, 1694: 1, 852: 4, 1464: 2, 2223: 1, 605: 16, 2411: 1, 687: 10, 1525: 2, 721: 13, 473: 31, 1152: 1, 2148: 1, 1732: 2, 1057: 3, 1241: 2, 1602: 2, 1188: 1, 706: 14, 2281: 1, 2052: 2, 677: 11, 1021: 3, 1379: 1, 1165: 3, 1054: 3, 1648: 1, 2502: 1, 1012: 4, 1390: 1, 1353: 2, 1504: 1, 1044: 3, 928: 7, 1389: 4, 1254: 1, 1465: 3, 2280: 2, 1137: 3, 2880: 1, 1385: 1, 766: 13, 905: 6, 1213: 2, 2070: 1, 1395: 1, 1180: 4, 1305: 1, 1142: 3, 2059: 1, 1402: 2, 1005: 5, 2337: 1, 800: 7, 1310: 2, 1572: 2, 1999: 1, 1327: 3, 1375: 1, 3123: 1, 952: 5, 1300: 3, 1719: 1, 2578: 1, 1064: 5, 1629: 3, 937: 4, 1050: 6, 1787: 2, 733: 9, 1178: 3, 1024: 3, 1185: 4, 921: 4, 838: 7, 2253: 1, 1115: 2, 2238: 2, 2153: 1, 908: 6, 1135: 4, 931: 6, 2795: 1, 732: 11, 1352: 2, 1523: 1, 1460: 2, 1127: 4, 791: 6, 875: 9, 1419: 2, 1349: 2, 835: 5, 858: 1, 2072: 1, 1773: 1, 1751: 2, 2607: 1, 988: 4, 1712: 3, 917: 5, 812: 2, 1424: 2, 959: 5, 1229: 2, 3128: 1, 757: 6, 807: 3, 2344: 1, 1140: 3, 1083: 2, 1125: 2, 893: 6, 1260: 2, 1803: 1, 1147: 2, 1799: 2, 1497: 2, 983: 5, 1363: 2, 1444: 4, 1191: 4, 2216: 1, 1257: 5, 1475: 1, 3102: 1, 554: 15, 1333: 2, 650: 18, 1753: 1, 1722: 1, 3142: 1, 1004: 1, 1168: 7, 1923: 2, 773: 9, 1156: 5, 14691: 1, 2966: 1, 1272: 2, 1149: 4, 711: 15, 2370: 1, 709: 8, 1374: 1, 1834: 1, 1033: 6, 842: 6, 1947: 1, 868: 7, 2043: 1, 1977: 1, 1821: 1, 923: 4, 1788: 1, 857: 4, 741: 15, 1592: 2, 1043: 3, 1798: 2, 1684: 1, 1321: 2, 1348: 1, 1123: 3, 1154: 3, 1007: 2, 1714: 1, 569: 23, 1076: 2, 1248: 4, 895: 6, 839: 4, 2151: 1, 944: 4, 1733: 1, 1856: 1, 1087: 4, 1252: 1, 1711: 2, 1435: 1, 2986: 1, 1580: 2, 1440: 1, 2145: 2, 2419: 1, 1089: 5, 1001: 5, 4607: 1, 1097: 2, 1433: 1, 1650: 1, 738: 7, 950: 3, 1240: 2, 1669: 1, 647: 12, 1073: 2, 1998: 1, 1034: 3, 2647: 1, 880: 2, 967: 2, 1242: 2, 957: 3, 1397: 2, 1785: 1, 1750: 2, 2096: 1, 1040: 4, 1313: 2, 1274: 2, 3272: 1, 2861: 1, 1279: 3, 1265: 2, 1940: 1, 1326: 3, 1972: 1, 1270: 1, 1228: 1, 862: 4, 1430: 1, 1519: 1, 1015: 4, 1554: 1, 3161: 1, 1247: 4, 1423: 3, 1221: 1, 1543: 1, 1429: 2, 1640: 1, 1237: 3, 1143: 2, 1357: 2, 806: 6, 1330: 1, 2249: 1, 1539: 1, 969: 6, 837: 7, 2397: 1, 1194: 2, 912: 5, 3146: 1, 938: 1, 913: 5, 1605: 2, 1775: 1, 2168: 2, 2000: 2, 1576: 1, 2758: 1, 1045: 2, 1068: 4, 2378: 1, 1120: 1, 1615: 2, 1139: 3, 1463: 2, 966: 6, 1245: 2, 2824: 1, 2501: 1, 1451: 2, 1417: 1, 929: 3, 970: 1, 2570: 1, 1271: 2, 2792: 1, 2300: 1, 2622: 1, 1362: 3, 861: 4, 975: 4, 924: 4, 1317: 2, 1091: 5, 1528: 1, 2420: 1, 3994: 1, 1246: 1, 1383: 1, 1132: 1, 956: 2, 971: 4, 1492: 1, 1915: 2, 2512: 2, 1410: 1, 871: 7, 4359: 1, 3499: 1, 1224: 3, 1392: 2, 1689: 2, 2162: 1, 973: 2, 1591: 1, 994: 1, 1022: 2, 1710: 1, 849: 6, 1175: 2, 1234: 2, 2464: 1, 848: 3, 1285: 1, 2087: 1, 1067: 2, 1921: 1, 2092: 1, 1110: 5, 2434: 1, 2174: 1, 2093: 2, 1527: 1, 2406: 1, 1144: 2, 2775: 1, 1807: 1, 1505: 1, 2374: 1, 1312: 2, 1308: 2, 1530: 1, 1959: 1, 1910: 1, 2753: 1, 1408: 1, 870: 3, 1350: 1, 1136: 2, 1852: 2, 1145: 1, 1456: 1, 2240: 1, 1653: 1, 2593: 1, 1897: 1, 2333: 1, 1713: 1, 1216: 1, 1631: 1, 1939: 1, 1008: 3, 1679: 2, 1201: 1, 3218: 1, 939: 3, 2660: 1, 2919: 1, 1244: 1, 1620: 2, 1848: 1, 1227: 2, 2155: 1, 1359: 2, 1210: 2, 1200: 3, 765: 4, 1565: 1, 2388: 1, 996: 2, 1167: 1, 1437: 1, 1704: 1, 1046: 5, 1582: 1, 1936: 1, 1196: 1, 2996: 1, 1480: 1, 1337: 2, 1193: 4, 2056: 1, 1853: 1, 856: 3, 1590: 2, 1970: 1, 1643: 1, 2742: 1, 1965: 2, 2212: 1, 2263: 1, 1526: 1, 3075: 1, 1173: 2, 942: 5, 2266: 2, 1544: 1, 813: 2, 1315: 2, 1502: 1, 1051: 3, 1996: 1, 1468: 1, 1920: 2, 990: 3, 2916: 1, 1483: 3, 1517: 2, 1226: 1, 1495: 1, 1882: 1, 1121: 2, 1288: 2, 833: 5, 1212: 1, 1869: 1, 1808: 2, 1282: 1, 1895: 1, 1347: 1, 1701: 2, 1219: 2, 1438: 1, 1630: 1, 1883: 2, 2581: 1, 1567: 1, 1324: 2, 1378: 1, 1659: 2, 1284: 3, 1103: 2, 0: 20, 4810: 1, 1422: 1, 1908: 1, 1681: 1, 777: 7, 1276: 3, 2009: 1, 1784: 1, 2849: 1, 1593: 1, 2423: 1, 3122: 1, 1304: 3, 1027: 1, 1339: 1, 1442: 1, 1747: 1, 2219: 1, 997: 2, 1296: 1, 2275: 1, 1813: 1, 2614: 1, 2114: 1, 1183: 2, 1644: 1, 930: 2, 1791: 1, 1718: 1, 1256: 2, 1190: 1, 2227: 1, 1550: 2, 1663: 1, 1405: 1, 1377: 1, 1583: 1, 2296: 1, 3158: 1, 1222: 1, 1618: 1, 1665: 1, 1211: 1, 982: 3, 1386: 1, 1205: 1, 1490: 1, 2208: 1, 1062: 2, 2540: 1, 1162: 1, 1499: 1, 1093: 2, 1249: 1, 1399: 1, 1953: 1, 1471: 1, 1851: 1, 2003: 1, 1299: 1, 1514: 1, 2134: 2, 968: 2, 2527: 1, 1662: 1, 1116: 3, 934: 1, 1868: 1, 3806: 1, 1129: 1, 2141: 1, 1098: 1, 3026: 1, 1612: 2, 1645: 1, 1341: 1, 1450: 1, 2276: 1, 1683: 1, 1278: 1, 1269: 1, 1238: 1, 2462: 1, 1342: 1, 1559: 2, 1268: 1, 1579: 1, 1124: 1, 2963: 1, 1624: 2, 1937: 1, 2837: 1, 1166: 1, 1159: 1, 5307: 1, 4128: 1, 2206: 1, 2555: 1, 1035: 1, 1455: 1, 1571: 1, 1253: 1, 1746: 1, 1922: 1, 1606: 1, 15686: 1, 2123: 1, 1323: 1, 2244: 1, 1398: 1, 1307: 1, 1770: 1, 2676: 1, 1862: 1, 2446: 1, 1956: 1, 2993: 1, 1466: 1, 1391: 1, 2048: 1, 1343: 1, 1138: 1, 1806: 1, 2391: 1}
    #reviewCountDic = dict(sorted(reviewCountDic.items(), key=lambda x: x[1], reverse=True))

    #plt.bar(list(reviewCountDic.keys()), reviewCountDic.values(), color='g')
    #plt.bar(list(reviewCountDic.keys()), reviewCountDic.values(), color='g')
    hist_list = [key for key, val in reviewCountDic.items() for _ in range(val)]
    # plt.hist(hist_list, color='g', bins=[1, 2, 3, 4, 5, 10, 15, 20, 50, 100])
    bins=[1,  5, 10, 15, 20, 50, 100, 200, 500, 1000, 2000]
    hist, bin_edges = np.histogram(hist_list, bins) # make the histogram

    fig,ax = plt.subplots()

    # Plot the histogram heights against integers on the x axis
    ax.bar(range(len(hist)),hist,width=1) 
    ax.set_xticks([i-0.5 for i,j in enumerate(hist)])
    ax.set_xticklabels(['{}'.format(bins[i]) for i,j in enumerate(hist)])
    plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
    plt.xlabel("Number of reviews per user")
    plt.ylabel("Frequency")
    plt.title("Number of reviews per user and corresponding frequency")

    plt.show()

def reviews_per_business():
    businessDF = parse_json(business_path)
    reviewCountList = businessDF['review_count'].to_numpy()
    stateCountList = businessDF['state'].to_numpy()
    # reviewCountList = reviewCountList[np.where(stateCountList == 'MA')]
    bins=[1, 5, 10, 15, 20, 50, 100, 200, 500, 1000, 2000]
    hist, bin_edges = np.histogram(reviewCountList, bins) # make the histogram

    fig,ax = plt.subplots()

    # Plot the histogram heights against integers on the x axis
    ax.bar(range(len(hist)),hist,width=1) 
    ax.set_xticks([i-0.5 for i,j in enumerate(hist)])
    ax.set_xticklabels(['{}'.format(bins[i]) for i,j in enumerate(hist)])
    plt.xlabel("Number of reviews per business")
    plt.ylabel("Frequency")
    plt.title("Number of reviews per business and corresponding frequency")

    plt.show()

# reviews_per_business()
# reviews_per_user()

def reviews_per_state():
    businessDF = parse_json(business_path)
    businessDF = businessDF[~businessDF['state'].isin(['DE', 'MN', 'VA', 'KY', 'ABE', 'KS', 'NC', 'OK', 'ON', 'NH', 'NM', 'AL', 'HI', 'WI', 'ME', 'WY', 'AZ'])]
    stateCountList = businessDF['state'].to_numpy()
    letter_counts = Counter(stateCountList)
    df = pd.DataFrame.from_dict(letter_counts, orient='index')
    df.plot(kind='bar', legend=False)
    plt.xlabel("State")
    plt.ylabel("Number of businesses")
    plt.title("Number of businesses per state")
    plt.show()

# TODO: for plot 1, color restaurants by categories

# old 150 5 0.01 0.2
# new 100 15 0.02 0.2
def generate_embeddings():
  businessDF = parse_json(business_path)
  with open('GA_restaurants_indices.json') as f:
    GA_restaurant_dict = json.load(f)
  GA_ID2Rest_dict = {}
  for key in GA_restaurant_dict:
    GA_ID2Rest_dict[GA_restaurant_dict[key]] = key

  businessDF = businessDF[businessDF['business_id'].isin(GA_restaurant_dict)]
  reviewDF = pd.read_csv("GA.csv")
  reviewBusinessIDs = reviewDF['business_id'].tolist()
  reviewBusinessIDSet = set(reviewBusinessIDs)
  reviewBusinessIDs = list(reviewBusinessIDSet)
  restaurantCount = len(reviewBusinessIDs)
  randomList = random.sample(range(0,restaurantCount), 30)
  for i, index in enumerate(randomList):
    randomList[i] = reviewBusinessIDs[index]

  reader = Reader(rating_scale=(1, 5))
  totalDataset = Dataset.load_from_df(reviewDF[['user_id', 'business_id', 'stars']], reader)
  trainset = totalDataset.build_full_trainset()
  # trainset, testset = train_test_split(totalDataset, test_size=.1)
  algo = SVD(n_factors = 150, n_epochs = 15, lr_all = 0.01, reg_all = 0.02)
  algo.fit(trainset)

  # predictions = algo.test(testset)
  # accuracy.rmse(predictions)

  tsne = TSNE(n_components=2, n_iter=500, verbose=3, random_state=6)
  businessEmbedding = tsne.fit_transform(algo.qi)
  embeddingDF = pd.DataFrame(columns=['x', 'y'], data=businessEmbedding)
  # plot 1
  # embeddingDF.plot.scatter('x', 'y')
  # plt.show()

  # plot 2
  modelIndexList = []
  subsetDF = pd.DataFrame(columns=['x', 'y', 'title'])
  for index in randomList:
    modelIndexList.append(trainset.to_inner_iid(index))
  for i, index in enumerate(modelIndexList):
    row = embeddingDF.iloc[index]
    new_row = {'x': row['x'], 'y': row['y'], 'title': businessDF[businessDF['business_id'] == GA_ID2Rest_dict[randomList[i]]]['name']}
    subsetDF = subsetDF.append(new_row, ignore_index=True)
  fig = px.scatter(
      subsetDF, x='x', y='y', text='title',
      )
  fig.show()
  report = dp.Report(dp.Plot(fig) ) #Create a report
  report.publish(name='Yelp Embedding 2', open=True, visibility='PUBLIC')