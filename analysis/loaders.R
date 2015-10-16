load.ing1.analysis <- function(path, rates, w_ies, js){
  df <- NULL
  for(r_s in rates){
    for(w_ie in w_ies){
      for(j in js){
        try({
          # rate-{1}_ie-{2}_j-{3}
          name <- paste("rate-", as.character(r_s),
                        "_ie-", as.character(w_ie),
                        "_j-", as.character(j),
                        "_analysis.csv", sep="")
          dat <- read.csv(paste(path, name, sep=""), header=FALSE)
          colnames(dat) <- c("stat", "x")
          dat["rate"] <- rep(r_s, nrow(dat))
          dat["w_ie"] <- rep(w_ie, nrow(dat))
          dat["j"] <- rep(j, nrow(dat))
          df <- rbind(df, dat)
        })
      }
    }
  }
  df
}

load.ping1.analysis <- function(path, rates, w_eis, w_ies, js){
  df <- NULL
  for(r_s in rates){
    for(w_ie in w_ies){
      for(w_ei in w_eis){
        for(j in js){
          try({
            #             rate-5_ei-1.0_ie-0.2_j-13_analysis
            name <- paste("rate-", as.character(r_s),
                          "_ei-", as.character(w_ei),
                          "_ie-", as.character(w_ie),
                          "_j-", as.character(j),
                          "_analysis.csv", sep="")
            dat <- read.csv(paste(path, name, sep=""), header=FALSE)
            colnames(dat) <- c("stat", "x")
            dat["rate"] <- rep(r_s, nrow(dat))
            dat["w_ei"] <- rep(w_ei, nrow(dat))
            dat["w_ie"] <- rep(w_ie, nrow(dat))
            dat["j"] <- rep(j, nrow(dat))
            df <- rbind(df, dat)
          })
        }
      }
    }
  }
  df
}

load.async1.analysis <- function(path, rates, js){
  df <- NULL
  for(r_s in rates){
    for(j in js){
      try({
        # rate-{1}_ie-{2}_j-{3}
        name <- paste("rate-", as.character(r_s),
                      "_j-", as.character(j),
                      "_analysis.csv", sep="")
        dat <- read.csv(paste(path, name, sep=""), header=FALSE)
        colnames(dat) <- c("stat", "x")
        dat["rate"] <- rep(r_s, nrow(dat))
        dat["w_ei"] <- rep("0.1", nrow(dat))
        dat["w_ie"] <- rep("0.5", nrow(dat))
        dat["j"] <- rep(j, nrow(dat))
        df <- rbind(df, dat)
      })
    }
  }
  df
}

load.ing1.poprates <- function(path, pop, rates, w_ies, js){
  df <- NULL
  for(r_s in rates){
    for(w_ie in w_ies){
      poprates <- NULL
      for(j in js){
        try({
          # rate-{1}_ie-{2}_j-{3}
          name <- paste("rate-", as.character(r_s),
                        "_ie-", as.character(w_ie),
                        "_j-", as.character(j),
                        "_poprates_", as.character(pop),
                        ".csv", sep="")
          dat <- NULL
          dat <- read.csv(paste(path, name, sep=""), header=FALSE)
          colnames(dat) <- c("sample_time", "poprate")
          poprates <- cbind(poprates, dat$poprate)
        })
      }
      df <- rbind(df,
                  data.frame(
                    poprate = rowMeans(poprates),
                    sample_time = dat$sample_time,
                    rate = rep(r_s, nrow(poprates)),
                    w_ie = rep(w_ie, nrow(poprates))
                  )
      )
    }
  }
  df
}

load.ing2.analysis <- function(path, I_es, js){
  df <- NULL
  for(I_e in I_es){
    for(j in js){
      try({
        name <- paste("I_e-", as.character(I_e), "-", as.character(I_e),
                      "_j-", as.character(j),
                      "_analysis.csv", sep="")
        dat <- read.csv(paste(path, name, sep=""), header=FALSE)
        colnames(dat) <- c("stat", "x")
        dat["I_e"] <- rep(I_e, nrow(dat))
        dat["j"] <- rep(j, nrow(dat))
        dat["w_ie"] <- rep("0.5", nrow(dat))
        df <- rbind(df, dat)
      })
    }
  }
  df
}

load.ing3.analysis <- function(path, I_i_sigmas, js){
  df <- NULL
  for(I_i in I_i_sigmas){
    for(j in js){
      try({
        name <- paste("I_i_sigma-", as.character(I_i),
                      "_j-", as.character(j),
                      "_analysis.csv", sep="")
        dat <- read.csv(paste(path, name, sep=""), header=FALSE)
        colnames(dat) <- c("stat", "x")
        dat["I_i_sigma"] <- rep(I_i, nrow(dat))
        dat["I_e"] <- rep("0.25", nrow(dat))
        dat["j"] <- rep(j, nrow(dat))
        dat["w_ie"] <- rep("0.5", nrow(dat))
        df <- rbind(df, dat)
      })
    }
  }
  df
}
