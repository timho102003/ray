import { Color, createStyles, makeStyles } from "@material-ui/core";
import { blue, blueGrey, cyan, green, red } from "@material-ui/core/colors";
import { CSSProperties } from "@material-ui/core/styles/withStyles";
import classNames from "classnames";
import React, { ReactNode } from "react";
import { TaskStatus } from "../pages/job/hook/useJobProgress";
import { ActorEnum } from "../type/actor";
import { JobStatus } from "../type/job";
import { PlacementGroupState } from "../type/placementGroup";
import {
  ServeApplicationStatus,
  ServeDeploymentStatus,
  ServeHTTPProxyStatus,
  ServeReplicaState,
} from "../type/serve";

const orange = "#DB6D00";
const grey = "#5F6469";

const colorMap = {
  node: {
    ALIVE: green,
    DEAD: red,
  },
  worker: {
    ALIVE: green,
    DEAD: red,
  },
  actor: {
    [ActorEnum.ALIVE]: green,
    [ActorEnum.DEAD]: red,
    [ActorEnum.DEPENDENCIES_UNREADY]: orange,
    [ActorEnum.PENDING_CREATION]: orange,
    [ActorEnum.RESTARTING]: orange,
  },
  task: {
    [TaskStatus.FAILED]: red,
    [TaskStatus.FINISHED]: green,
    [TaskStatus.RUNNING]: blue,
    [TaskStatus.SUBMITTED_TO_WORKER]: orange,
    [TaskStatus.PENDING_NODE_ASSIGNMENT]: orange,
    [TaskStatus.PENDING_ARGS_AVAIL]: orange,
    [TaskStatus.UNKNOWN]: grey,
  },
  job: {
    [JobStatus.PENDING]: orange,
    [JobStatus.RUNNING]: blue,
    [JobStatus.STOPPED]: grey,
    [JobStatus.SUCCEEDED]: green,
    [JobStatus.FAILED]: red,
  },
  placementGroup: {
    [PlacementGroupState.PENDING]: orange,
    [PlacementGroupState.CREATED]: green,
    [PlacementGroupState.REMOVED]: grey,
    [PlacementGroupState.RESCHEDULING]: orange,
  },
  serveApplication: {
    [ServeApplicationStatus.NOT_STARTED]: grey,
    [ServeApplicationStatus.DEPLOYING]: orange,
    [ServeApplicationStatus.RUNNING]: green,
    [ServeApplicationStatus.DEPLOY_FAILED]: red,
    [ServeApplicationStatus.DELETING]: orange,
  },
  serveDeployment: {
    [ServeDeploymentStatus.UPDATING]: orange,
    [ServeDeploymentStatus.HEALTHY]: green,
    [ServeDeploymentStatus.UNHEALTHY]: red,
  },
  serveReplica: {
    [ServeReplicaState.STARTING]: orange,
    [ServeReplicaState.UPDATING]: orange,
    [ServeReplicaState.RECOVERING]: orange,
    [ServeReplicaState.RUNNING]: green,
    [ServeReplicaState.STOPPING]: red,
  },
  serveHttpProxy: {
    [ServeHTTPProxyStatus.HEALTHY]: green,
    [ServeHTTPProxyStatus.UNHEALTHY]: red,
    [ServeHTTPProxyStatus.STARTING]: orange,
  },
} as {
  [key: string]: {
    [key: string]: Color | string;
  };
};

const typeMap = {
  deps: blue,
  INFO: cyan,
  ERROR: red,
} as {
  [key: string]: Color;
};

const useStyles = makeStyles((theme) =>
  createStyles({
    root: {
      padding: "2px 8px",
      border: "solid 1px",
      borderRadius: 4,
      fontSize: 12,
      margin: 2,
      display: "inline-flex",
      alignItems: "center",
    },
    afterIcon: {
      marginLeft: 4,
    },
  }),
);

export const StatusChip = ({
  type,
  status,
  suffix,
  icon,
}: {
  type: string;
  status: string | ActorEnum | ReactNode;
  suffix?: string;
  icon?: ReactNode;
}) => {
  const classes = useStyles();
  let color: Color | string = blueGrey;

  if (typeMap[type]) {
    color = typeMap[type];
  } else if (
    typeof status === "string" &&
    colorMap[type] &&
    colorMap[type][status]
  ) {
    color = colorMap[type][status];
  }

  const colorValue = typeof color === "string" ? color : color[500];

  const style: CSSProperties = {};
  style.color = colorValue;
  style.borderColor = colorValue;
  if (color !== blueGrey) {
    style.backgroundColor = `${colorValue}20`;
  }

  return (
    <span className={classes.root} style={style}>
      {icon}
      <span className={classNames({ [classes.afterIcon]: icon !== undefined })}>
        {status}
      </span>
      {suffix}
    </span>
  );
};
