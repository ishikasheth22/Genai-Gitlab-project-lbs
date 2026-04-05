---
title: "Business Continuity Plan"
controlled_document: true
tags:
  - security_policy
  - security_policy_cpir
---

{{< label name="Visibility: Audit" color="#E24329" >}}

## Purpose

Business Continuity Plan is the process involved in creating a system of prevention and recovery from potential threats to GitLab. The plan ensures that personnel and assets are protected and are able to function quickly in the event of a business disruption.

## Scope

GitLab, by its remote-only nature, is not easily affected by typical causes of business disruption, such as local failures of equipment, power supplies, telecommunications, social unrest, terrorist attacks, fire, or natural disasters. System data from the [Business Impact Analysis](/handbook/security/security-assurance/security-risk/storm-program/business-impact-analysis/) may be leveraged as part of business continuity planning and testing.

## Roles & Responsibilities

| Role | Responsibility |
|-----------|-----------|
| GitLab Team Members | Responsible for following the requirements in this procedure |
| Security Risk Engineer | Responsible for conducting periodic Business Impact Analysis and applying Critical System Tiers to Tech Stack Systems |
| Security Risk Senior Engineer | Responsible for integrating bcp considerations into the IRP tabletop exercise. |
| Security Risk Manager | Responsible for reviewing and operationalizing procedures BCP procedures. Serves as back up for Senior Engineer if necessary. |

## Procedure

### BCP for Remote Workers

In case of an all-remote company like GitLab, it is sufficient to have simple contingency plans in the form of service-level agreements with companies that host our data and services. The advantage of an all-remote workforce like GitLab is that if there are clusters of people or systems that are unavailable, the rest of the company will continue to operate normally. By nature of GitLab’s structure as a fully remote company, the Incident Response Process represents the bulk of our Business Continuity Plan. More information regarding Incident Response and its relationship with Business Continuity [can be found here](/handbook/business-technology/entapps-documentation/policies/gitlab-business-continuity-plan/).

### Escalations

[Procedures can be found here](/handbook/security/security-operations/sirt/sec-incident-response/#internal-engagement--escalation-for-high-severity-incidents) in the event that an incident needs to be escalated within GitLab. SIRT maintains an [on-call schedule](/handbook/security/security-operations/sirt/engaging-security-on-call/) to ensure ongoing coverage and cross-training for incident response procedures.

Individual system outages should be escalated to system owners per the [Tech Stack](https://gitlab.com/gitlab-com/www-gitlab-com/-/blob/master/data/tech_stack.yml).

### What triggers the Business Continuity Plan

The Business Continuity Plan is triggered during [SIRT's Incident Response Procedures]((/handbook/security/security-operations/sirt/sec-incident-response/) in the event that support is required outside of SIRT's standard operations.

Upon activation SIRT will ping the #BCP_Events channel for triage.

BCP for individual system outages are managed by system owners per the [Tech Stack](https://gitlab.com/gitlab-com/www-gitlab-com/-/blob/master/data/tech_stack.yml).

#### Engagement message template

When the BCP is triggered due to a security incident, a thread will be opened in #BCP_Events tagging all members using the template below.

> @here, GitLab's Business Continuity Plan has been invoked in response to a SIRT incident. Tagged stakeholders should review incident details to determine whether action is necessary.

| Business Impact | Handbook Page | Slack Channel |
| --- | --- | --- |
| Is there a financial impact? | [Handbook Page](/handbook/finance/) | #finance |
| Is there a personnel impact? | [Handbook Page](/handbook/people-group/people-ops-tech-analytics/) | #peopleops-eng |
| Is there an infrastructure impact? | [Handbook Page](/handbook/engineering/infrastructure/) | #infrastructure-lounge |
| Is there a legal impact? | [Handbook Page](/handbook/legal/) | #legal |
| Is there a system impact | N/A - [Tech Stack](https://gitlab.com/gitlab-com/www-gitlab-com/-/blob/master/data/tech_stack.yml) ownership | N/A - Tech Stack technical_owner |

### Vendor communication and service restoration plan

System owners are responsible for maintaining vendor communication plans as part of Business Continuity preparation. These plans ensure all systems and services return to normal operations following the outage, establish communication protocols with service providers for coordination, and identify alternate suppliers for critical services when applicable.

### Root Cause Analysis

Any time the business continuity plan is activated, a root cause analysis should be performed to identify lessons-learned. The root cause analysis should review the trigger of the event and recommend remediations that prevent future occurrences of the issue. Additionally, if opportunities for improvements in the response to the specific business continuity scenario are identified, the business continuity plan and applicable procedures should be updated to reflect those lessons learned.

### External Communications

External communications should be issued once the scope and impact of the incident have been determined. [Procedures can be found here](/handbook/security/security-operations/sirt/security-incident-communication-plan/#communicating-externally).

## Business Continuity Test

After formalizing the business continuity plan, or BCP,  the next important step is to test the plan. Testing verifies the effectiveness of the plan, informs plan participants on what to do in a real scenario, and identifies areas where the plan needs to be strengthened. A test of the plan must be conducted at least annually, and should have clearly defined test objectives and success criteria prior to the test.

Note that by nature of GitLab's operating structure as a fully remote company, the Incident Response Process encompasses the bulk of our Business Continuity Plan. The scope of  Incident Response testing may be expanded to include other parties that could be impacted by the incident and serves as evidence of our BCP testing activity.

### Testing the plan

Testing can present a lot of challenges and requires time and resource investment. With that in mind, it may make more sense to conduct a tabletop test at a conference room rather than involving the entire organization in a full-blown drill. An initial "dry run" of the plan can be performed by conducting a structured walk-through test of the BCP. The initial testing is done in sections and after normal business hours to minimize disruptions. Subsequent tests can occur during normal business hours. An actual test-run can be performed eventually. The various types of tests that can be conducted include checklist tests, simulation tests, parallel tests, and full interruption tests.

### Requirements and Considerations

The below criteria must be met when performing the BCP test:

- Tests should include an assessment of financial, operational and reputational impacts of disruptions.
- Testing should include engagement of key vendors, suppliers, and partners when necessary.
- Metrics should be in place to gauge effectiveness of the plan and identify weaknesses.
- Test results must be formally documented, including gaps and weaknesses identified during testing.
- Plans should be developed to address any gaps and weaknesses, with the plan updated to document any changes.

### Business Continuity Plan Testing Scenarios

There are several types of tests, such as a plan review, a tabletop test, or a simulation test, which was detailed in the previous section.
Some example testing scenarios that may be performed, are provided below:

1. Data Loss/Breach

    - One of the most prevalent workplace disasters today. Cause of data loss or breach could be due to any of the following:
        - Ransomware and cyberattacks
        - Unintentionally erased files or folders
        - Server/drive crash
        - Datacenter outage
    - Data is mission-critical and losing it can have many serious consequences, such as significantly impacting sales and logistics applications.
    - The goal is to regain access to that data as soon as possible. Restoring backup is the solution. However, who's responsible for that? What's the communication plan in this case? What are the priorities? Who needs to be contacted right away? Are there any vendors involved? These and many other questions will be answered during this test.

1. Data Recovery Testing

    - This testing scenario, is used to make sure that the backup and recovery  systems work as intended. To prove that, run a test that involves losing a bulk of data, and then try to recover it.
    - Some of the elements to be evaluated include the RTO, and whether the team met its objectives.
    - Also make note of, if there were any damage to the files during recovery? Was the backup stored in the cloud, recovered successfully and on time.

1. Emergency Communication

    - Being able to communicate during a disaster or an emergency is crucial. Yet, the most disruptive events can leave with no traditional means of staying in contact.
    - For these scenarios, the BC plan needs to outline the actions to be taken. An alternate mode of communication should be tested for its reliability and efficiency, for a company like GitLab which has team members all around the globe.
    - Regular updates to all GitLab team members contact information, so that all of them can receive timely notifications thus streamlining the disaster scenario process.

1. Individual System Outage

    - System owners should have plans to ensure key operational functions remain viable in the event of a system outage. This includes understanding key system service level agreements, escalation paths and backup operations, if required.

This plan is reviewed and updated annually or after significant organizational changes.

## Exceptions

Exceptions to this procedure will be tracked as per the [Information Security Policy Exception Management Process](/handbook/security/controlled-document-procedure/#exceptions).

## References

- [Business Impact Analysis](/handbook/security/security-assurance/security-risk/storm-program/business-impact-analysis/)
- [Disaster Recovery Plan (DRP)](https://gitlab.com/gitlab-com/gl-infra/readiness/-/blob/master/library/disaster-recovery/index.md)
- [Security Risk Team](/handbook/security/security-assurance/security-risk/)
- [SIRT](/handbook/security/security-operations/sirt/)
