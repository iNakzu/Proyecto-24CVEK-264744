/**********************************************************************
 Copyright (c) 2020-2024, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/

#pragma once

#include "unitree_lidar_sdk.h"

using namespace unilidar_sdk2;

void run(UnitreeLidarReader *lreader){

    std::string versionSDK, versionHardware, versionFirmware;
    while (!lreader->getVersionOfLidarFirmware(versionFirmware))
    {
        lreader->runParse();
    }
    lreader->getVersionOfLidarHardware(versionHardware);
    lreader->getVersionOfSDK(versionSDK);

    float dirtyPercentage;
    while(!lreader->getDirtyPercentage(dirtyPercentage)){
        lreader->runParse();
    }
    
    double timeDelay;
    while(!lreader->getTimeDelay(timeDelay)){
        lreader->runParse();
    }

    int result;
    LidarImuData imu;
    PointCloudUnitree cloud;
    while (true)
    {
        result = lreader->runParse();

        switch (result)
        {
        case LIDAR_IMU_DATA_PACKET_TYPE:
            if (lreader->getImuData(imu))
            {
            }
            break;

        case LIDAR_POINT_DATA_PACKET_TYPE:
            if (lreader->getPointCloud(cloud))
            {
                for (const auto& point : cloud.points)
                {
                    printf("(%f, %f, %f, %f, %f, %d)\n",
                           point.x,
                           point.y,
                           point.z,
                           point.intensity,
                           point.time,
                           point.ring);
                }
            }

            break;

        default:
            break;
        }

    }

}