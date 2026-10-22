/**
 * Jenkins API Client for TrainerMon
 * Connects to Jenkins on http://localhost:8089 with root:root
 * Manages CSRF Crumb, Session Cookies, Job Provisioning, Build Triggering, and Log Streaming
 */

export interface JenkinsCrumbSession {
  crumbRequestField: string;
  crumb: string;
  cookie: string;
}

export interface JenkinsBuildInfo {
  id: string;
  number: number;
  building: boolean;
  result: "SUCCESS" | "FAILURE" | "UNSTABLE" | "ABORTED" | null;
  duration: number;
  estimatedDuration: number;
  timestamp: number;
  url: string;
}

export class JenkinsClient {
  private baseUrl: string;
  private authHeader: string;
  private cachedSession: JenkinsCrumbSession | null = null;
  private sessionTimestamp: number = 0;

  constructor(baseUrl: string = "http://localhost:8089", username: string = "root", apiTokenOrPass: string = "root") {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.authHeader = "Basic " + Buffer.from(`${username}:${apiTokenOrPass}`).toString("base64");
  }

  /**
   * Fetches or re-uses valid Jenkins CSRF crumb and matching JSESSIONID cookie
   */
  async getCrumbAndSession(): Promise<JenkinsCrumbSession> {
    const now = Date.now();
    if (this.cachedSession && now - this.sessionTimestamp < 1800000) {
      return this.cachedSession;
    }

    const res = await fetch(`${this.baseUrl}/crumbIssuer/api/json`, {
      headers: {
        Authorization: this.authHeader,
      },
    });

    if (!res.ok) {
      throw new Error(`Failed to obtain Jenkins CSRF crumb (HTTP ${res.status}: ${res.statusText})`);
    }

    const setCookie = res.headers.get("set-cookie") || "";
    // Extract first cookie or JSESSIONID
    const cookie = setCookie.split(";")[0] || "";
    const data = await res.json();

    this.cachedSession = {
      crumbRequestField: data.crumbRequestField || "Jenkins-Crumb",
      crumb: data.crumb,
      cookie,
    };
    this.sessionTimestamp = now;
    return this.cachedSession;
  }

  /**
   * Checks if a pipeline job exists
   */
  async jobExists(jobName: string): Promise<boolean> {
    try {
      const res = await fetch(`${this.baseUrl}/job/${encodeURIComponent(jobName)}/api/json`, {
        headers: {
          Authorization: this.authHeader,
        },
      });
      return res.status === 200;
    } catch {
      return false;
    }
  }

  /**
   * Generates Jenkins XML configuration for a Pipeline job bound to the Git repository
   */
  private generatePipelineXml(repoUrl: string, branch: string = "main"): string {
    return `<?xml version='1.1' encoding='UTF-8'?>
<flow-definition plugin="workflow-job">
  <description>TrainerMon CI/CD Pipeline - Autonomous Pretraining, Quality Gate &amp; Self-Healing</description>
  <keepDependencies>false</keepDependencies>
  <properties>
    <org.jenkinsci.plugins.workflow.job.properties.DisableConcurrentBuildsJobProperty/>
  </properties>
  <definition class="org.jenkinsci.plugins.workflow.cps.CpsScmFlowDefinition" plugin="workflow-cps">
    <scm class="hudson.plugins.git.GitSCM" plugin="git">
      <configVersion>2</configVersion>
      <userRemoteConfigs>
        <hudson.plugins.git.UserRemoteConfig>
          <url>${repoUrl}</url>
        </hudson.plugins.git.UserRemoteConfig>
      </userRemoteConfigs>
      <branches>
        <hudson.plugins.git.BranchSpec>
          <name>*/${branch}</name>
        </hudson.plugins.git.BranchSpec>
      </branches>
      <doGenerateSubmoduleConfigurations>false</doGenerateSubmoduleConfigurations>
      <submoduleCfg class="empty-list"/>
      <extensions/>
    </scm>
    <scriptPath>Jenkinsfile</scriptPath>
    <lightweight>true</lightweight>
  </definition>
  <triggers>
    <hudson.triggers.SCMTrigger>
      <spec>H/2 * * * *</spec>
      <ignorePostCommitHooks>false</ignorePostCommitHooks>
    </hudson.triggers.SCMTrigger>
  </triggers>
  <disabled>false</disabled>
</flow-definition>`;
  }

  /**
   * Provisions or updates the pipeline job
   */
  async ensurePipelineJob(jobName: string, repoUrl: string, branch: string = "main"): Promise<{ created: boolean; updated: boolean }> {
    const session = await this.getCrumbAndSession();
    const xml = this.generatePipelineXml(repoUrl, branch);
    const exists = await this.jobExists(jobName);

    const headers: Record<string, string> = {
      Authorization: this.authHeader,
      "Content-Type": "application/xml",
      [session.crumbRequestField]: session.crumb,
    };
    if (session.cookie) {
      headers["Cookie"] = session.cookie;
    }

    if (!exists) {
      const res = await fetch(`${this.baseUrl}/createItem?name=${encodeURIComponent(jobName)}`, {
        method: "POST",
        headers,
        body: xml,
      });
      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Failed to create Jenkins pipeline job: ${errText}`);
      }
      return { created: true, updated: false };
    } else {
      const res = await fetch(`${this.baseUrl}/job/${encodeURIComponent(jobName)}/config.xml`, {
        method: "POST",
        headers,
        body: xml,
      });
      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Failed to update Jenkins pipeline job: ${errText}`);
      }
      return { created: false, updated: true };
    }
  }

  /**
   * Triggers an immediate build execution
   */
  async triggerBuild(jobName: string): Promise<boolean> {
    const session = await this.getCrumbAndSession();
    const headers: Record<string, string> = {
      Authorization: this.authHeader,
      [session.crumbRequestField]: session.crumb,
    };
    if (session.cookie) {
      headers["Cookie"] = session.cookie;
    }

    const res = await fetch(`${this.baseUrl}/job/${encodeURIComponent(jobName)}/build`, {
      method: "POST",
      headers,
    });
    return res.status === 201 || res.status === 200;
  }

  /**
   * Retrieves latest build information
   */
  async getLatestBuild(jobName: string): Promise<JenkinsBuildInfo | null> {
    try {
      const res = await fetch(`${this.baseUrl}/job/${encodeURIComponent(jobName)}/lastBuild/api/json`, {
        headers: {
          Authorization: this.authHeader,
        },
      });
      if (!res.ok) return null;
      const data = await res.json();
      return {
        id: data.id,
        number: data.number,
        building: data.building,
        result: data.result,
        duration: data.duration,
        estimatedDuration: data.estimatedDuration,
        timestamp: data.timestamp,
        url: data.url,
      };
    } catch {
      return null;
    }
  }

  /**
   * Retrieves raw console logs for a specific build or the last build
   */
  async getConsoleOutput(jobName: string, buildNumber: number | string = "lastBuild"): Promise<string> {
    try {
      const res = await fetch(`${this.baseUrl}/job/${encodeURIComponent(jobName)}/${buildNumber}/consoleText`, {
        headers: {
          Authorization: this.authHeader,
        },
      });
      if (!res.ok) return "";
      return await res.text();
    } catch {
      return "";
    }
  }
}
