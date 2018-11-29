import Dependencies._

lazy val root = (project in file(".")).
  settings(
    inThisBuild(List(
      organization := "com.example",
      scalaVersion := "2.12.7",
      version      := "0.1.0-SNAPSHOT"
    )),
    name := "neural_network",
    libraryDependencies ++= Seq(
      "be.botkop" %% "numsca" % "0.1.4",
      "com.typesafe.scala-logging" %% "scala-logging" % "3.9.0",
      scalaTest % Test
    )
  )
