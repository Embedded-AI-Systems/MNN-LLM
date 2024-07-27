//
//  calib3d_test.cpp
//  MNN
//
//  Created by MNN on 2022/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <gtest/gtest.h>
#include <opencv2/calib3d.hpp>
#include <MNN/AutoTime.hpp>
#include "cv/calib3d.hpp"
#include "test_env.hpp"

#ifdef MNN_CALIB3D_TEST

static Env<float> testEnv(img_name, true);

// solvePnP
TEST(solvePnP, base) {
    float model_points[18] = {
        0.0, 0.0, 0.0, 0.0, -330.0, -65.0, -225.0, 170.0, -135.0,
        225.0, 170.0, -135.0, -150.0, -150.0, -125.0, 150.0, -150.0, -125.0
    };
    float image_points[12] = {
        359, 391, 399, 561, 337, 297, 513, 301, 345, 465, 453, 469
    };
    float camera_matrix[9] = {
        1200, 0, 600, 0, 1200, 337.5, 0, 0, 1
    };
    float dist_coeffs[4] = { 0, 0, 0, 0 };
    VARP mnnObj = _Const(model_points, {6, 3});
    VARP mnnImg = _Const(image_points, {6, 2});
    VARP mnnCam = _Const(camera_matrix, {3, 3});
    VARP mnnCoe = _Const(dist_coeffs, {4, 1});
    
    std::pair<VARP, VARP> mnnRes = solvePnP(mnnObj, mnnImg, mnnCam, mnnCoe);
    cv::Mat cvObj = cv::Mat(6, 3, CV_32F, model_points);
    cv::Mat cvImg = cv::Mat(6, 2, CV_32F, image_points);
    cv::Mat cvCam = cv::Mat(3, 3, CV_32F, camera_matrix);
    cv::Mat cvCoe = cv::Mat(4, 1, CV_32F, dist_coeffs);
    std::vector<float> rv(3), tv(3);
    cv::Mat rvecs(rv),tvecs(tv);
    cv::solvePnP(cvObj, cvImg, cvCam, cvCoe, rvecs, tvecs, false, cv::SOLVEPNP_SQPNP);
    EXPECT_TRUE(testEnv.equal(rvecs, mnnRes.first) && testEnv.equal(tvecs, mnnRes.second));
}

TEST(solvePnP, shoe1) {
    float model_points[96] = {
        -0.030799999833106995, 0.0007699999841861427, -0.004399999976158142, -0.02474999986588955, 0.01209999993443489, -0.004399999976158142, -0.01759999990463257, 0.01924999989569187, -0.004399999976158142, -0.008799999952316284, 0.021449999883770943, -0.004399999976158142, 0.0, 0.022549999877810478, -0.004399999976158142, 0.008799999952316284, 0.021449999883770943, -0.004399999976158142, 0.01759999990463257, 0.01924999989569187, -0.004399999976158142, 0.02474999986588955, 0.01209999993443489, -0.004399999976158142, 0.030799999833106995, 0.0007699999841861427, -0.004399999976158142, 0.02474999986588955, -0.010999999940395355, -0.004399999976158142, 0.01759999990463257, -0.016499999910593033, -0.004399999976158142, 0.008799999952316284, -0.01979999989271164, -0.004399999976158142, 0.0, -0.021449999883770943, -0.004399999976158142, -0.008799999952316284, -0.01979999989271164, -0.004399999976158142, -0.01759999990463257, -0.016499999910593033, -0.004399999976158142, -0.02474999986588955, -0.010999999940395355, -0.004399999976158142, -0.030799999833106995, 0.0007699999841861427, 0.004399999976158142, -0.02474999986588955, 0.01209999993443489, 0.004399999976158142, -0.01759999990463257, 0.01924999989569187, 0.004399999976158142, -0.008799999952316284, 0.021449999883770943, 0.004399999976158142, 0.0, 0.022549999877810478, 0.004399999976158142, 0.008799999952316284, 0.021449999883770943, 0.004399999976158142, 0.01759999990463257, 0.01924999989569187, 0.004399999976158142, 0.02474999986588955, 0.01209999993443489, 0.004399999976158142, 0.030799999833106995, 0.0007699999841861427, 0.004399999976158142, 0.02474999986588955, -0.010999999940395355, 0.004399999976158142, 0.01759999990463257, -0.016499999910593033, 0.004399999976158142, 0.008799999952316284, -0.01979999989271164, 0.004399999976158142, 0.0, -0.021449999883770943, 0.004399999976158142, -0.008799999952316284, -0.01979999989271164, 0.004399999976158142, -0.01759999990463257, -0.016499999910593033, 0.004399999976158142, -0.02474999986588955, -0.010999999940395355, 0.004399999976158142
    };
    float image_points[64] = {
        -0.0013057183241471648, 0.09299874305725098, -0.010455395095050335, 0.07131724804639816, -0.026523245498538017, 0.046155307441949844, -0.04385042563080788, 0.02129918523132801, -0.06332534551620483, -0.004408818203955889, -0.08762498944997787, -0.03180401027202606, -0.11104003340005875, -0.05363212525844574, -0.11999830603599548, -0.05565139278769493, -0.11811254918575287, -0.042330302298069, -0.10884306579828262, -0.023449651896953583, -0.09400629997253418, -0.003091213759034872, -0.07834033668041229, 0.017524732276797295, -0.06170866638422012, 0.038823917508125305, -0.04244353622198105, 0.06119990721344948, -0.023676535114645958, 0.082001693546772, -0.005143166519701481, 0.09810793399810791, -0.01900496520102024, 0.10677685588598251, -0.028071928769350052, 0.08504459261894226, -0.04472069442272186, 0.060138411819934845, -0.06216795742511749, 0.03575018048286438, -0.08179453015327454, 0.010372052900493145, -0.1054963767528534, -0.017218351364135742, -0.1289127767086029, -0.039570920169353485, -0.13818585872650146, -0.04231918603181839, -0.13637131452560425, -0.029076160863041878, -0.12689357995986938, -0.009866510517895222, -0.1117326021194458, 0.011127307079732418, -0.09655033051967621, 0.03239363059401512, -0.07965713739395142, 0.053755879402160645, -0.06071771681308746, 0.07591498643159866, -0.04190481826663017, 0.09673473984003067, -0.02356795221567154, 0.11175438016653061
    };
    float camera_matrix[9] = {
       1, 0, 0, 0, 1, 0, 0, 0, 1
    };
    float dist_coeffs[4] = { 0, 0, 0, 0 };
    VARP mnnObj = _Const(model_points, {32, 3});
    VARP mnnImg = _Const(image_points, {32, 2});
    VARP mnnCam = _Const(camera_matrix, {3, 3});
    VARP mnnCoe = _Const(dist_coeffs, {4, 1});
    std::pair<VARP, VARP> mnnRes = solvePnP(mnnObj, mnnImg, mnnCam, mnnCoe);
    cv::Mat cvObj = cv::Mat(32, 3, CV_32F, model_points);
    cv::Mat cvImg = cv::Mat(32, 2, CV_32F, image_points);
    cv::Mat cvCam = cv::Mat(3, 3, CV_32F, camera_matrix);
    cv::Mat cvCoe = cv::Mat(4, 1, CV_32F, dist_coeffs);
    std::vector<float> rv(3), tv(3);
    cv::Mat rvecs(rv),tvecs(tv);
    cv::solvePnP(cvObj, cvImg, cvCam, cvCoe, rvecs, tvecs, false, cv::SOLVEPNP_SQPNP);
    EXPECT_TRUE(testEnv.equal(rvecs, mnnRes.first) && testEnv.equal(tvecs, mnnRes.second));
}

TEST(solvePnP, shoe2) {
    float model_points[96] = {
        -0.030799999833106995, 0.0007699999841861427, -0.004399999976158142, -0.02474999986588955, 0.01209999993443489, -0.004399999976158142, -0.01759999990463257, 0.01924999989569187, -0.004399999976158142, -0.008799999952316284, 0.021449999883770943, -0.004399999976158142, 0.0, 0.022549999877810478, -0.004399999976158142, 0.008799999952316284, 0.021449999883770943, -0.004399999976158142, 0.01759999990463257, 0.01924999989569187, -0.004399999976158142, 0.02474999986588955, 0.01209999993443489, -0.004399999976158142, 0.030799999833106995, 0.0007699999841861427, -0.004399999976158142, 0.02474999986588955, -0.010999999940395355, -0.004399999976158142, 0.01759999990463257, -0.016499999910593033, -0.004399999976158142, 0.008799999952316284, -0.01979999989271164, -0.004399999976158142, 0.0, -0.021449999883770943, -0.004399999976158142, -0.008799999952316284, -0.01979999989271164, -0.004399999976158142, -0.01759999990463257, -0.016499999910593033, -0.004399999976158142, -0.02474999986588955, -0.010999999940395355, -0.004399999976158142, -0.030799999833106995, 0.0007699999841861427, 0.004399999976158142, -0.02474999986588955, 0.01209999993443489, 0.004399999976158142, -0.01759999990463257, 0.01924999989569187, 0.004399999976158142, -0.008799999952316284, 0.021449999883770943, 0.004399999976158142, 0.0, 0.022549999877810478, 0.004399999976158142, 0.008799999952316284, 0.021449999883770943, 0.004399999976158142, 0.01759999990463257, 0.01924999989569187, 0.004399999976158142, 0.02474999986588955, 0.01209999993443489, 0.004399999976158142, 0.030799999833106995, 0.0007699999841861427, 0.004399999976158142, 0.02474999986588955, -0.010999999940395355, 0.004399999976158142, 0.01759999990463257, -0.016499999910593033, 0.004399999976158142, 0.008799999952316284, -0.01979999989271164, 0.004399999976158142, 0.0, -0.021449999883770943, 0.004399999976158142, -0.008799999952316284, -0.01979999989271164, 0.004399999976158142, -0.01759999990463257, -0.016499999910593033, 0.004399999976158142, -0.02474999986588955, -0.010999999940395355, 0.004399999976158142
    };
    float image_points[64] = {
        -0.028867509216070175, 0.20026835799217224, -0.01353165041655302, 0.17049874365329742, 0.007216880097985268, 0.13621856272220612, 0.031573839485645294, 0.10374260693788528, 0.05863713100552559, 0.07126667350530624, 0.08930887281894684, 0.03879072144627571, 0.11907847970724106, 0.01623797044157982, 0.12629535794258118, 0.023454850539565086, 0.11817637085914612, 0.04600759968161583, 0.10284051299095154, 0.0703645572066307, 0.08479831367731094, 0.09652574360370636, 0.06404978781938553, 0.12358903139829636, 0.04059493914246559, 0.1515544354915619, 0.01714007928967476, 0.17951983213424683, -0.0063147698529064655, 0.20207257568836212, -0.027965400367975235, 0.21650633215904236, -0.003608440048992634, 0.21921266615390778, 0.011727429926395416, 0.18944303691387177, 0.032475948333740234, 0.1560649871826172, 0.05683290958404541, 0.12358903139829636, 0.08299410343170166, 0.09111308306455612, 0.11366582661867142, 0.05863713100552559, 0.14343544840812683, 0.03518227860331535, 0.15065231919288635, 0.04149705171585083, 0.1425333321094513, 0.06404978781938553, 0.12809957563877106, 0.08930887281894684, 0.11005737632513046, 0.11637215316295624, 0.08930887281894684, 0.14433754980564117, 0.06495189666748047, 0.1723029613494873, 0.04149705171585083, 0.1993662416934967, 0.01804219000041485, 0.22101688385009766, -0.003608440048992634, 0.23454852402210236
    };
    float camera_matrix[9] = {
       1, 0, 0, 0, 1, 0, 0, 0, 1
    };
    float dist_coeffs[4] = { 0, 0, 0, 0 };
    VARP mnnObj = _Const(model_points, {32, 3});
    VARP mnnImg = _Const(image_points, {32, 2});
    VARP mnnCam = _Const(camera_matrix, {3, 3});
    VARP mnnCoe = _Const(dist_coeffs, {4, 1});
    std::pair<VARP, VARP> mnnRes = solvePnP(mnnObj, mnnImg, mnnCam, mnnCoe);
    cv::Mat cvObj = cv::Mat(32, 3, CV_32F, model_points);
    cv::Mat cvImg = cv::Mat(32, 2, CV_32F, image_points);
    cv::Mat cvCam = cv::Mat(3, 3, CV_32F, camera_matrix);
    cv::Mat cvCoe = cv::Mat(4, 1, CV_32F, dist_coeffs);
    std::vector<float> rv(3), tv(3);
    cv::Mat rvecs(rv),tvecs(tv);
    cv::solvePnP(cvObj, cvImg, cvCam, cvCoe, rvecs, tvecs, false, cv::SOLVEPNP_SQPNP);
    EXPECT_TRUE(testEnv.equal(rvecs, mnnRes.first) && testEnv.equal(tvecs, mnnRes.second));
}
TEST(solvePnP, shoe3) {
    float model_points[96] = {
        0.030799999833106995, 0.0007699999841861427, -0.004399999976158142, 0.02474999986588955, 0.01209999993443489, -0.004399999976158142, 0.01759999990463257, 0.01924999989569187, -0.004399999976158142, 0.008799999952316284, 0.021449999883770943, -0.004399999976158142, 0.0, 0.022549999877810478, -0.004399999976158142, -0.008799999952316284, 0.021449999883770943, -0.004399999976158142, -0.01759999990463257, 0.01924999989569187, -0.004399999976158142, -0.02474999986588955, 0.01209999993443489, -0.004399999976158142, -0.030799999833106995, 0.0007699999841861427, -0.004399999976158142, -0.02474999986588955, -0.010999999940395355, -0.004399999976158142, -0.01759999990463257, -0.016499999910593033, -0.004399999976158142, -0.008799999952316284, -0.01979999989271164, -0.004399999976158142, 0.0, -0.021449999883770943, -0.004399999976158142, 0.008799999952316284, -0.01979999989271164, -0.004399999976158142, 0.01759999990463257, -0.016499999910593033, -0.004399999976158142, 0.02474999986588955, -0.010999999940395355, -0.004399999976158142, 0.030799999833106995, 0.0007699999841861427, 0.004399999976158142, 0.02474999986588955, 0.01209999993443489, 0.004399999976158142, 0.01759999990463257, 0.01924999989569187, 0.004399999976158142, 0.008799999952316284, 0.021449999883770943, 0.004399999976158142, 0.0, 0.022549999877810478, 0.004399999976158142, -0.008799999952316284, 0.021449999883770943, 0.004399999976158142, -0.01759999990463257, 0.01924999989569187, 0.004399999976158142, -0.02474999986588955, 0.01209999993443489, 0.004399999976158142, -0.030799999833106995, 0.0007699999841861427, 0.004399999976158142, -0.02474999986588955, -0.010999999940395355, 0.004399999976158142, -0.01759999990463257, -0.016499999910593033, 0.004399999976158142, -0.008799999952316284, -0.01979999989271164, 0.004399999976158142, 0.0, -0.021449999883770943, 0.004399999976158142, 0.008799999952316284, -0.01979999989271164, 0.004399999976158142, 0.01759999990463257, -0.016499999910593033, 0.004399999976158142, 0.02474999986588955, -0.010999999940395355, 0.004399999976158142
    };
    float image_points[64] = {
        0.10013417899608612, 0.2760455906391144, 0.07667932659387589, 0.22823376953601837, 0.03608439117670059, 0.17049874365329742, -0.011727427132427692, 0.11366582661867142, -0.06495190411806107, 0.05683291330933571, -0.12990380823612213, -0.0009021097212098539, -0.19214937090873718, -0.04510548710823059, -0.2201147824525833, -0.05232236534357071, -0.2174084484577179, -0.028867511078715324, -0.19214937090873718, 0.018944304436445236, -0.15335865318775177, 0.06856033951044083, -0.10825317353010178, 0.11907848715782166, -0.06134346127510071, 0.167792409658432, -0.010825317353010178, 0.21921266615390778, 0.03969283029437065, 0.2607097029685974, 0.08660253882408142, 0.2895772159099579, 0.06134346127510071, 0.30040255188941956, 0.03879071772098541, 0.2534928321838379, -0.0027063293382525444, 0.19575782120227814, -0.05051814764738083, 0.13892489671707153, -0.10284051299095154, 0.083896204829216, -0.16598819196224213, 0.026161182671785355, -0.22733165323734283, -0.018944304436445236, -0.25710126757621765, -0.027965402230620384, -0.25529706478118896, -0.005412658676505089, -0.2309400886297226, 0.04420337826013565, -0.19214937090873718, 0.09562363475561142, -0.14794600009918213, 0.1452396661043167, -0.10013417899608612, 0.19395360350608826, -0.04871392622590065, 0.24356962740421295, 0.0018042194424197078, 0.2832624614238739, 0.04871392622590065, 0.3121299743652344
    };
    float camera_matrix[9] = {
       1, 0, 0, 0, 1, 0, 0, 0, 1
    };
    float dist_coeffs[4] = { 0, 0, 0, 0 };
    VARP mnnObj = _Const(model_points, {32, 3});
    VARP mnnImg = _Const(image_points, {32, 2});
    VARP mnnCam = _Const(camera_matrix, {3, 3});
    VARP mnnCoe = _Const(dist_coeffs, {4, 1});
    std::pair<VARP, VARP> mnnRes = solvePnP(mnnObj, mnnImg, mnnCam, mnnCoe);
    cv::Mat cvObj = cv::Mat(32, 3, CV_32F, model_points);
    cv::Mat cvImg = cv::Mat(32, 2, CV_32F, image_points);
    cv::Mat cvCam = cv::Mat(3, 3, CV_32F, camera_matrix);
    cv::Mat cvCoe = cv::Mat(4, 1, CV_32F, dist_coeffs);
    std::vector<float> rv(3), tv(3);
    cv::Mat rvecs(rv),tvecs(tv);
    cv::solvePnP(cvObj, cvImg, cvCam, cvCoe, rvecs, tvecs, false, cv::SOLVEPNP_SQPNP);
    EXPECT_TRUE(testEnv.equal(rvecs, mnnRes.first) && testEnv.equal(tvecs, mnnRes.second));
}
TEST(solvePnP, shoe4) {
    float model_points[96] = {
        -0.030799999833106995, 0.0007699999841861427, -0.004399999976158142, -0.02474999986588955, 0.01209999993443489, -0.004399999976158142, -0.01759999990463257, 0.01924999989569187, -0.004399999976158142, -0.008799999952316284, 0.021449999883770943, -0.004399999976158142, 0.0, 0.022549999877810478, -0.004399999976158142, 0.008799999952316284, 0.021449999883770943, -0.004399999976158142, 0.01759999990463257, 0.01924999989569187, -0.004399999976158142, 0.02474999986588955, 0.01209999993443489, -0.004399999976158142, 0.030799999833106995, 0.0007699999841861427, -0.004399999976158142, 0.02474999986588955, -0.010999999940395355, -0.004399999976158142, 0.01759999990463257, -0.016499999910593033, -0.004399999976158142, 0.008799999952316284, -0.01979999989271164, -0.004399999976158142, 0.0, -0.021449999883770943, -0.004399999976158142, -0.008799999952316284, -0.01979999989271164, -0.004399999976158142, -0.01759999990463257, -0.016499999910593033, -0.004399999976158142, -0.02474999986588955, -0.010999999940395355, -0.004399999976158142, -0.030799999833106995, 0.0007699999841861427, 0.004399999976158142, -0.02474999986588955, 0.01209999993443489, 0.004399999976158142, -0.01759999990463257, 0.01924999989569187, 0.004399999976158142, -0.008799999952316284, 0.021449999883770943, 0.004399999976158142, 0.0, 0.022549999877810478, 0.004399999976158142, 0.008799999952316284, 0.021449999883770943, 0.004399999976158142, 0.01759999990463257, 0.01924999989569187, 0.004399999976158142, 0.02474999986588955, 0.01209999993443489, 0.004399999976158142, 0.030799999833106995, 0.0007699999841861427, 0.004399999976158142, 0.02474999986588955, -0.010999999940395355, 0.004399999976158142, 0.01759999990463257, -0.016499999910593033, 0.004399999976158142, 0.008799999952316284, -0.01979999989271164, 0.004399999976158142, 0.0, -0.021449999883770943, 0.004399999976158142, -0.008799999952316284, -0.01979999989271164, 0.004399999976158142, -0.01759999990463257, -0.016499999910593033, 0.004399999976158142, -0.02474999986588955, -0.010999999940395355, 0.004399999976158142
    };
    float image_points[64] = {
        0.20568102598190308, -0.04690970852971077, 0.20297469198703766, -0.07667932659387589, 0.19665992259979248, -0.09832996129989624, 0.18493250012397766, -0.1064489483833313, 0.17320507764816284, -0.10284051299095154, 0.1632818579673767, -0.08660253882408142, 0.1560649871826172, -0.06404979526996613, 0.15245655179023743, -0.03337806090712547, 0.14794600009918213, -0.0036084388848394156, 0.149750217795372, 0.022552743554115295, 0.15335865318775177, 0.04059493914246559, 0.16147764027118683, 0.04871392622590065, 0.17140084505081177, 0.04781181737780571, 0.18403038382530212, 0.03518227860331535, 0.19395360350608826, 0.012629536911845207, 0.20207259058952332, -0.014433755539357662, 0.17861773073673248, -0.04330126941204071, 0.17591139674186707, -0.074875108897686, 0.1695966273546219, -0.0974278524518013, 0.1587713211774826, -0.10554683953523636, 0.14794600009918213, -0.1019384041428566, 0.1380227953195572, -0.08570042252540588, 0.12990380823612213, -0.06134346127510071, 0.1271974742412567, -0.030671730637550354, 0.12358903884887695, 0.0, 0.12539325654506683, 0.027063293382525444, 0.12990380823612213, 0.04510548710823059, 0.13712067902088165, 0.05322447419166565, 0.1470438838005066, 0.05232236534357071, 0.1587713211774826, 0.03788860887289047, 0.167792409658432, 0.015335865318775177, 0.17591139674186707, -0.012629536911845207
    };
    float camera_matrix[9] = {
       1, 0, 0, 0, 1, 0, 0, 0, 1
    };
    float dist_coeffs[4] = { 0, 0, 0, 0 };
    VARP mnnObj = _Const(model_points, {32, 3});
    VARP mnnImg = _Const(image_points, {32, 2});
    VARP mnnCam = _Const(camera_matrix, {3, 3});
    VARP mnnCoe = _Const(dist_coeffs, {4, 1});
    std::pair<VARP, VARP> mnnRes = solvePnP(mnnObj, mnnImg, mnnCam, mnnCoe);
    cv::Mat cvObj = cv::Mat(32, 3, CV_32F, model_points);
    cv::Mat cvImg = cv::Mat(32, 2, CV_32F, image_points);
    cv::Mat cvCam = cv::Mat(3, 3, CV_32F, camera_matrix);
    cv::Mat cvCoe = cv::Mat(4, 1, CV_32F, dist_coeffs);
    std::vector<float> rv(3), tv(3);
    cv::Mat rvecs(rv),tvecs(tv);
    cv::solvePnP(cvObj, cvImg, cvCam, cvCoe, rvecs, tvecs, false, cv::SOLVEPNP_SQPNP);
    EXPECT_TRUE(testEnv.equal(rvecs, mnnRes.first) && testEnv.equal(tvecs, mnnRes.second));
}
#endif
