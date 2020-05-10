function get_means(n_classes, r)
    step = 2π/n_classes
    ϕs = collect(0:step:2π)[1:end-1]
    mean_x = reshape(cos.(ϕs) * r, (1,n_classes))
    mean_y = reshape(sin.(ϕs) * r, (1,n_classes))
    means = vcat(mean_x, mean_y)
    return means
end

function make_moons_ssl()
    Knet.seed!(2020)
    Random.seed!(2020)
    n_samples = 1000
    data = MLJ.make_moons(n_samples;noise=0.05)
    data = convert(atype, permutedims(hcat(data[1][1], data[1][2])))
    labels = convert(atype, ones(1,n_samples)) * (-1)
    idx1 = [2 3 6 19 33]
    labels[idx1] .= 1
    idx0 = [1 4 8 9 35]
    labels[idx0] .= 0
    means = [-3.5 3.5; -3.5 3.5]
    return data, labels, means
end

function make_circles_ssl()
    Knet.seed!(2020)
    Random.seed!(2020)
    n_samples = 1000
    data = MLJ.make_circles(n_samples; noise=0.05, factor=0.4)
    data = convert(atype, permutedims(hcat(data[1][1], data[1][2])))
    labels = convert(atype, ones(1,n_samples)) * (-1)
    idx1 = [1 4 7 21]
    labels[idx1] .= 1
    idx0 = [2 3 6 16 26 28 33]
    labels[idx0] .= 0
    r = 3.5
    n_classes = 2
    means = get_means(n_classes, r)
    return data, labels, means
end

function make_8gauss_ssl()
    eight_gauss = npzread("toy_datasets/8gauss.npz")
    data = convert(atype, permutedims(eight_gauss["data"]))
    labels = convert(atype, permutedims(eight_gauss["labels"]))
    r = 5.5
    n_classes = 3
    means = get_means(n_classes, r)
    return data, labels, means
end

function make_pinwheel_ssl()
    pinwheel = npzread("toy_datasets/pinwheel.npz")
    data = convert(atype, permutedims(pinwheel["data"]))
    labels = convert(atype, permutedims(pinwheel["labels"]))
    r = 5.5
    n_classes = 5
    means = get_means(n_classes, r)
    return data, labels, means
end

function make_miniboone_ssl()
    #train_data training data with 50 features (50x65000)
    #train_labels: every datapoint is unlabeled, except for 20 points (10 from each class) (1x65000)
    #train_gt_labels: ground truth labels for training datapoints, can be used in calculating train accuracy (1x65000)
    #test_data: test data (50x7998)
    #test_labels: ground truth labels for test data points (1x7998)
    #means: 10*per class average of labeled train_data (50x2)
    #Note: labels are (0,1)
    
    miniboone = convert(Matrix, CSV.read("uci_datasets/miniboone_dataset.csv", header=0))
    samples_per_class = 32500
    Random.seed!(246)
    class_0_samples = Distributions.sample(1:36499, 36499, replace=false)
    class_1_samples = Distributions.sample(36500:130064, 93565, replace=false)
    train_class_0 = class_0_samples[1:samples_per_class]
    train_class_1 = class_1_samples[1:samples_per_class]
    class_0 = convert(atype, permutedims(miniboone[train_class_0, :]))
    class_1 = convert(atype, permutedims(miniboone[train_class_1, :]))
    
    train_data = hcat(class_0, class_1)
    train_gt_labels = hcat(convert(atype, zeros(1,samples_per_class)), convert(atype, ones(1,samples_per_class)))
    
    permutation = Distributions.sample(1:2*samples_per_class, 2*samples_per_class, replace=false)
    
    train_data = train_data[:, permutation]
    train_gt_labels = train_gt_labels[:,permutation]
    
    train_labels = convert(atype, ones(1,65000)) * (-1)
    idx0 = [2 4 5 6 7 8 9 11 16 20]
    idx1 = [1 3 10 12 13 14 15 17 18 19]
    train_labels[idx0] .= 0
    train_labels[idx1] .= 1
    
    test_class_0 = class_0_samples[samples_per_class+1:36499]
    test_class_1 = class_1_samples[samples_per_class+1:samples_per_class+3999]
    class_0 = convert(atype, permutedims(miniboone[test_class_0, :]))
    class_1 = convert(atype, permutedims(miniboone[test_class_1, :]))
    test_data = hcat(class_0, class_1)
    
    test_labels = hcat(convert(atype, zeros(1,3999)), convert(atype, ones(1,3999)))
    
    means =  zeros(50,2)
    for i in 1:50
        x = train_data[:,[2 4 5 6 7 8 9 11 16 20]]
        means[i,1] = sum(x[i,:])/10
    end
    for i in 1:50
        x = train_data[:,[1 3 10 12 13 14 15 17 18 19]]
        means[i,2] = sum(x[i,:])/10
    end
    means *= 10
    
    return train_data, train_labels, train_gt_labels, test_data, test_labels, means
    
end

function make_hepmass_ssl()
    #train_data: training data with 27 features (27x700000) 
    #train_labels: every datapoint is unlabeled, except for 20 points (10 from each class) (1x700000)
    #train_gt_labels: ground truth labels for training datapoints, can be used in calculating train accuracy (1x700000)
    #test_data: test data (27x350000)
    #test_labels: ground truth labeles for test datapoints (1x350000)
    #means: 10*per class average of labeled train_data (27x2)
    #NOTE: labels are (0,1)
    
    hepmass_train = convert(Matrix, CSV.read("uci_datasets/1000_train.csv"))
    hepmass_train = convert(atype, permutedims(hepmass_train))
    train_data = hepmass_train[2:28,:]
    train_gt_labels = reshape(hepmass_train[1,:], (1,700000))
    
    train_labels = convert(atype, ones(1,700000)) * (-1)
    idx0 = [3 5 7 8 9 10 13 15 16 18]
    idx1 = [1 2 4 6 11 12 14 17 19 21]
    train_labels[idx0] .= 0
    train_labels[idx1] .= 1
    
    hepmass_test = convert(Matrix, CSV.read("uci_datasets/1000_test.csv"))
    hepmass_test = convert(atype, permutedims(hepmass_test))
    test_data = hepmass_test[2:28,:]
    test_labels = reshape(hepmass_test[1,:], (1,350000))
    
    means =  zeros(27,2)
    for i in 1:27
        x = train_data[:,[3 5 7 8 9 10 13 15 16 18]]
        means[i,1] = sum(x[i,:])/10
    end
    for i in 1:27
        x = train_data[:,[1 2 4 6 11 12 14 17 19 21]]
        means[i,2] = sum(x[i,:])/10
    end
    means *= 10
    
    return train_data, train_labels, train_gt_labels, test_data, test_labels, means
end

function make_ag_news_ssl()
    #train_data: bert_embeddings of training data (768x120000) 
    #train_labels: every datapoint is unlabeled, except for 200 points (50 from each class) (1x120000)
    #train_gt_labels: ground truth labels for training datapoints, used in calculating train accuracy (1x120000)
    #test_data: bert embeddings of test data (768x7600)
    #test_labels: ground truth labeles for test datapoints (1x7600)
    #means: 10*per class average of labeled train_data (768x4)
    #NOTE: labels are (0,1,2,3)
    
    ag_train = npzread("nlp_datasets/ag_news_train.npz")
    train_gt_labels = convert(atype, permutedims(ag_train["labels"]))
    train_data = convert(atype, permutedims(ag_train["encodings"]))
    train_labels = convert(atype, ones(1,120000)) * (-1)
    idx0 = [3 6 7 8 9 14 16 21 22 32 34 39 43 48 53 54 55 58 64 65 68 70 72 74 78 82 89 98 112 115 116 119 122 125 126 128 134 139 145 146 150 152 156 158 165 166 192 193 199 202]
    idx1 = [1 2 17 24 25 26 27 29 30 31 33 36 44 45 46 47 52 60 61 62 63 66 69 71 77 81 87 90 91 92 93 97 99 103 106 111 113 117 124 130 132 136 137 143 144 149 153 155 159 160]
    idx2 = [4 10 11 13 18 23 35 37 41 49 51 67 76 79 84 88 104 105 107 109 110 114 123 127 129 131 133 138 140 148 151 154 157 164 169 179 180 182 187 189 191 195 196 197 198 201 204 206 209 210]
    idx3 = [5 12 15 19 20 28 38 40 42 50 56 57 59 73 75 80 83 85 86 94 95 96 100 101 102 108 118 120 121 135 141 142 147 162 167 170 174 175 176 177 181 185 186 188 194 205 215 220 223 228]
    train_labels[idx0] .= 0
    train_labels[idx1] .= 1
    train_labels[idx2] .= 2
    train_labels[idx3] .= 3
    
    means =  zeros(768,4)
    for i in 1:768
        x = train_data[:,[3, 6, 7, 8, 9, 14, 16, 21, 22, 32, 34, 39, 43, 48, 53, 54, 55, 58, 64, 65, 68, 70, 72, 74, 78, 82, 89, 98, 112, 115, 116, 119, 122, 125, 126, 128, 134, 139, 145, 146, 150, 152, 156, 158, 165, 166, 192, 193, 199, 202]]
        means[i,1] = sum(x[i,:])/50
    end
    for i in 1:768
        x = train_data[:,[1, 2, 17, 24, 25, 26, 27, 29, 30, 31, 33, 36, 44, 45, 46, 47, 52, 60, 61, 62, 63, 66, 69, 71, 77, 81, 87, 90, 91, 92, 93, 97, 99, 103, 106, 111, 113, 117, 124, 130, 132, 136, 137, 143, 144, 149, 153, 155, 159, 160]]
        means[i,2] = sum(x[i,:])/50
    end
    for i in 1:768
        x = train_data[:,[4, 10, 11, 13, 18, 23, 35, 37, 41, 49, 51, 67, 76, 79, 84, 88, 104, 105, 107, 109, 110, 114, 123, 127, 129, 131, 133, 138, 140, 148, 151, 154, 157, 164, 169, 179, 180, 182, 187, 189, 191, 195, 196, 197, 198, 201, 204, 206, 209, 210]]
        means[i,3] = sum(x[i,:])/50
    end
    for i in 1:768
        x = train_data[:,[5, 12, 15, 19, 20, 28, 38, 40, 42, 50, 56, 57, 59, 73, 75, 80, 83, 85, 86, 94, 95, 96, 100, 101, 102, 108, 118, 120, 121, 135, 141, 142, 147, 162, 167, 170, 174, 175, 176, 177, 181, 185, 186, 188, 194, 205, 215, 220, 223, 228]]
        means[i,4] = sum(x[i,:])/50
    end
    means *= 10
    
    ag_test = npzread("nlp_datasets/ag_news_test.npz")
    test_labels = convert(atype, permutedims(ag_test["labels"]))
    test_data = convert(atype, permutedims(ag_test["encodings"]))
    return train_data, train_labels, train_gt_labels, test_data, test_labels, means
end

function make_yahoo_answers_ssl()
    #train_data: bert_embeddings of training data (768x1400000) 
    #train_labels: every datapoint is unlabeled, except for 800 points (80 from each class) (1x1400000)
    #train_gt_labels: ground truth labels for training datapoints, can be used in calculating train accuracy (1x1400000)
    #test_data: bert embeddings of test data (768x60000)
    #test_labels: ground truth labeles for test datapoints (1x60000)
    #means: 10*per class average of labeled train_data (768x10)
    #NOTE: labels are (0,1,2,3,...,9)
    
    yahoo_train_labels = npzread("nlp_datasets/yahoo_answers_train_labels.npy")
    yahoo_train_labels[yahoo_train_labels.==-1] .= 9 #csv file da label 10 var, .train va .test filelarda olup label 0, npy file da olup -1
    train_gt_labels = convert(atype, permutedims(yahoo_train_labels))
    
    train_labels = convert(atype, ones(1,1400000)) * (-1)
    idx0 = [15 21 31 32 33 40 43 44 71 96 101 110 113 124 156 159 166 182 183 192 207 223 225 262 295 300 301 304 362 395 397 404 408 433 438 440 450 464 465 467 475 513 516 521 540 542 553 561 564 605 606 608 615 624 638 648 660 676 696 709 712 725 729 730 754 758 768 771 776 781 821 825 827 828 840 842 846 847 854 859 ]
    idx1 = [6 20 29 36 42 47 49 54 56 68 70 74 79 81 88 98 102 127 145 150 173 188 199 224 228 232 238 240 246 260 269 270 273 293 294 314 318 320 329 330 340 355 361 378 385 391 392 416 422 451 478 479 487 496 501 507 518 529 575 576 586 593 611 617 631 637 649 651 658 662 667 694 706 707 745 746 755 760 761 772 ]
    idx2 = [2 8 12 18 45 55 57 65 75 80 87 97 104 108 115 122 139 143 148 157 165 210 211 213 227 230 237 266 272 277 278 288 297 302 316 345 351 354 384 390 393 396 403 419 458 463 466 500 514 528 533 535 548 566 569 570 592 595 625 639 640 669 688 689 698 710 716 731 739 750 767 782 799 800 805 816 843 844 850 851 ]
    idx3 = [3 24 34 78 90 106 114 125 129 144 151 154 158 167 168 169 179 212 218 226 236 244 253 263 275 298 323 350 357 358 366 410 420 434 456 503 504 508 511 524 526 527 539 551 558 563 573 574 585 587 588 603 630 633 641 653 657 679 690 704 736 742 752 759 762 774 784 785 792 806 809 814 822 823 845 848 849 861 866 872 ]
    idx4 = [11 19 37 53 63 67 76 91 111 121 135 140 152 178 181 191 194 196 200 216 222 231 239 248 256 282 284 313 325 337 338 344 349 352 367 368 374 383 386 415 417 435 441 446 453 477 489 492 493 495 549 556 572 607 621 632 644 655 708 714 719 727 734 735 738 740 744 763 773 775 798 817 820 833 864 877 884 893 898 907 ]
    idx5 = [5 10 28 59 61 66 69 73 94 95 100 123 126 128 133 155 171 174 187 195 198 206 229 243 252 255 261 268 281 285 296 303 306 307 308 310 312 321 333 341 346 347 360 400 401 405 407 409 411 427 428 431 447 454 459 472 473 480 484 491 555 567 579 594 609 619 626 645 647 652 664 672 673 674 678 683 684 693 697 721 ]
    idx6 = [4 7 17 38 51 60 82 83 85 86 93 103 105 132 142 149 153 163 176 184 185 197 202 214 220 234 242 251 254 264 283 286 311 317 327 335 353 356 364 369 372 373 375 380 388 399 412 424 436 445 449 455 468 469 476 505 519 520 522 544 565 582 599 601 602 604 618 627 629 634 665 671 675 680 685 686 692 695 699 700 ]
    idx7 = [9 13 14 25 26 27 35 48 50 52 64 72 107 112 116 120 137 162 170 177 190 193 201 203 208 221 235 245 249 276 279 287 289 290 292 305 319 328 334 348 371 379 387 398 406 421 426 430 437 444 448 452 470 474 486 490 497 498 510 517 523 537 543 554 557 578 580 583 584 590 596 610 614 622 643 663 668 681 701 702 ]
    idx8 = [1 23 39 41 62 77 89 92 99 131 136 141 146 147 164 186 209 219 247 250 257 265 274 280 299 309 322 339 342 365 370 376 381 382 394 413 418 423 425 429 432 439 442 443 460 461 482 483 485 488 494 499 502 506 525 530 532 534 538 541 545 547 550 552 559 560 568 577 591 597 600 613 620 623 635 636 646 654 659 661 ]
    idx9 = [16 22 30 46 58 84 109 117 118 119 130 134 138 160 161 172 175 180 189 204 205 215 217 233 241 258 259 267 271 291 315 324 326 331 332 336 343 359 363 377 389 402 414 457 462 471 481 509 512 515 531 536 546 562 571 581 589 598 612 616 628 642 650 656 670 677 687 691 733 741 764 780 787 788 797 808 818 819 830 835 ]
    train_labels[idx0] .= 0
    train_labels[idx1] .= 1
    train_labels[idx2] .= 2
    train_labels[idx3] .= 3
    train_labels[idx4] .= 4
    train_labels[idx5] .= 5
    train_labels[idx6] .= 6
    train_labels[idx7] .= 7
    train_labels[idx8] .= 8
    train_labels[idx9] .= 9
    
    yahoo_train_encodings = npzread("nlp_datasets/yahoo_answers_train_encodings.npy")
    train_data = convert(atype, permutedims(yahoo_train_encodings))
    
    means =  zeros(768,10)
    for i in 1:768
        x = train_data[:,[15, 21, 31, 32, 33, 40, 43, 44, 71, 96, 101, 110, 113, 124, 156, 159, 166, 182, 183, 192, 207, 223, 225, 262, 295, 300, 301, 304, 362, 395, 397, 404, 408, 433, 438, 440, 450, 464, 465, 467, 475, 513, 516, 521, 540, 542, 553, 561, 564, 605, 606, 608, 615, 624, 638, 648, 660, 676, 696, 709, 712, 725, 729, 730, 754, 758, 768, 771, 776, 781, 821, 825, 827, 828, 840, 842, 846, 847, 854, 859 ]]
        means[i,1] = sum(x[i,:])/80
    end
    for i in 1:768
        x = train_data[:,[6, 20, 29, 36, 42, 47, 49, 54, 56, 68, 70, 74, 79, 81, 88, 98, 102, 127, 145, 150, 173, 188, 199, 224, 228, 232, 238, 240, 246, 260, 269, 270, 273, 293, 294, 314, 318, 320, 329, 330, 340, 355, 361, 378, 385, 391, 392, 416, 422, 451, 478, 479, 487, 496, 501, 507, 518, 529, 575, 576, 586, 593, 611, 617, 631, 637, 649, 651, 658, 662, 667, 694, 706, 707, 745, 746, 755, 760, 761, 772 ]]
        means[i,2] = sum(x[i,:])/80
    end
    for i in 1:768
        x = train_data[:,[2, 8, 12, 18, 45, 55, 57, 65, 75, 80, 87, 97, 104, 108, 115, 122, 139, 143, 148, 157, 165, 210, 211, 213, 227, 230, 237, 266, 272, 277, 278, 288, 297, 302, 316, 345, 351, 354, 384, 390, 393, 396, 403, 419, 458, 463, 466, 500, 514, 528, 533, 535, 548, 566, 569, 570, 592, 595, 625, 639, 640, 669, 688, 689, 698, 710, 716, 731, 739, 750, 767, 782, 799, 800, 805, 816, 843, 844, 850, 851 ]]
        means[i,3] = sum(x[i,:])/80
    end
    for i in 1:768
        x = train_data[:,[3, 24, 34, 78, 90, 106, 114, 125, 129, 144, 151, 154, 158, 167, 168, 169, 179, 212, 218, 226, 236, 244, 253, 263, 275, 298, 323, 350, 357, 358, 366, 410, 420, 434, 456, 503, 504, 508, 511, 524, 526, 527, 539, 551, 558, 563, 573, 574, 585, 587, 588, 603, 630, 633, 641, 653, 657, 679, 690, 704, 736, 742, 752, 759, 762, 774, 784, 785, 792, 806, 809, 814, 822, 823, 845, 848, 849, 861, 866, 872 ]]
        means[i,4] = sum(x[i,:])/80
    end
    for i in 1:768
        x = train_data[:,[11, 19, 37, 53, 63, 67, 76, 91, 111, 121, 135, 140, 152, 178, 181, 191, 194, 196, 200, 216, 222, 231, 239, 248, 256, 282, 284, 313, 325, 337, 338, 344, 349, 352, 367, 368, 374, 383, 386, 415, 417, 435, 441, 446, 453, 477, 489, 492, 493, 495, 549, 556, 572, 607, 621, 632, 644, 655, 708, 714, 719, 727, 734, 735, 738, 740, 744, 763, 773, 775, 798, 817, 820, 833, 864, 877, 884, 893, 898, 907 ]]
        means[i,5] = sum(x[i,:])/80
    end
    for i in 1:768
        x = train_data[:,[5, 10, 28, 59, 61, 66, 69, 73, 94, 95, 100, 123, 126, 128, 133, 155, 171, 174, 187, 195, 198, 206, 229, 243, 252, 255, 261, 268, 281, 285, 296, 303, 306, 307, 308, 310, 312, 321, 333, 341, 346, 347, 360, 400, 401, 405, 407, 409, 411, 427, 428, 431, 447, 454, 459, 472, 473, 480, 484, 491, 555, 567, 579, 594, 609, 619, 626, 645, 647, 652, 664, 672, 673, 674, 678, 683, 684, 693, 697, 721 ]]
        means[i,6] = sum(x[i,:])/80
    end
    for i in 1:768
        x = train_data[:,[4, 7, 17, 38, 51, 60, 82, 83, 85, 86, 93, 103, 105, 132, 142, 149, 153, 163, 176, 184, 185, 197, 202, 214, 220, 234, 242, 251, 254, 264, 283, 286, 311, 317, 327, 335, 353, 356, 364, 369, 372, 373, 375, 380, 388, 399, 412, 424, 436, 445, 449, 455, 468, 469, 476, 505, 519, 520, 522, 544, 565, 582, 599, 601, 602, 604, 618, 627, 629, 634, 665, 671, 675, 680, 685, 686, 692, 695, 699, 700 ]]
        means[i,7] = sum(x[i,:])/80
    end
    for i in 1:768
        x = train_data[:,[9, 13, 14, 25, 26, 27, 35, 48, 50, 52, 64, 72, 107, 112, 116, 120, 137, 162, 170, 177, 190, 193, 201, 203, 208, 221, 235, 245, 249, 276, 279, 287, 289, 290, 292, 305, 319, 328, 334, 348, 371, 379, 387, 398, 406, 421, 426, 430, 437, 444, 448, 452, 470, 474, 486, 490, 497, 498, 510, 517, 523, 537, 543, 554, 557, 578, 580, 583, 584, 590, 596, 610, 614, 622, 643, 663, 668, 681, 701, 702 ]]
        means[i,8] = sum(x[i,:])/80
    end
    for i in 1:768
        x = train_data[:,[1, 23, 39, 41, 62, 77, 89, 92, 99, 131, 136, 141, 146, 147, 164, 186, 209, 219, 247, 250, 257, 265, 274, 280, 299, 309, 322, 339, 342, 365, 370, 376, 381, 382, 394, 413, 418, 423, 425, 429, 432, 439, 442, 443, 460, 461, 482, 483, 485, 488, 494, 499, 502, 506, 525, 530, 532, 534, 538, 541, 545, 547, 550, 552, 559, 560, 568, 577, 591, 597, 600, 613, 620, 623, 635, 636, 646, 654, 659, 661 ]]
        means[i,9] = sum(x[i,:])/80
    end
    for i in 1:768
        x = train_data[:,[16, 22, 30, 46, 58, 84, 109, 117, 118, 119, 130, 134, 138, 160, 161, 172, 175, 180, 189, 204, 205, 215, 217, 233, 241, 258, 259, 267, 271, 291, 315, 324, 326, 331, 332, 336, 343, 359, 363, 377, 389, 402, 414, 457, 462, 471, 481, 509, 512, 515, 531, 536, 546, 562, 571, 581, 589, 598, 612, 616, 628, 642, 650, 656, 670, 677, 687, 691, 733, 741, 764, 780, 787, 788, 797, 808, 818, 819, 830, 835 ]]
        means[i,10] = sum(x[i,:])/80
    end
    means *= 10
    
    
    yahoo_test = npzread("nlp_datasets/yahoo_answers_test.npz")
    test_labels = yahoo_test["labels"]
    test_labels[test_labels.==-1] .= 9
    test_labels = convert(atype, permutedims(test_labels))
    
    test_data = convert(atype, permutedims(yahoo_test["encodings"]))
    return train_data, train_labels, train_gt_labels, test_data, test_labels, means
end

nothing
