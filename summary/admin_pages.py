from django.http import HttpResponseRedirect, HttpResponse, JsonResponse
from django.shortcuts import render
from .models import ApprovalModel, TranslationsModel, AdminModel, User_Model, SummaryModel, ContentModal


def admin_view_user_list(request):
    ReturnUser = list(User_Model.objects.values())
    return JsonResponse(ReturnUser, safe=False)


def admin_view_user(request, user_id):
    ReturnSingleUser = list(
        User_Model.objects.filter(user_id=user_id).values())
    return JsonResponse(ReturnSingleUser, safe=False)


def admin_view_single_request(request, req_id):
    userDetSend = userDet(request.session['UserName'])
    ReturnSingleUser = list(User_Model.objects.filter(user_id=req_id).values())
    return JsonResponse(ReturnSingleUser, safe=False)


def dashboard(request):
    context = {
        'pendingReq': ApprovalModel.objects.filter(approvalStatus='pending').count(),
        'SummariesDone': SummaryModel.objects.count(),
        'TransDone': TranslationsModel.objects.count(),
        'Adv_summary': SummaryModel.objects.count()-10,
        'ContentGen_CNT': ContentModal.objects.count()
    }
    return render(request, "admin_pages/dashboard.html", context)


def pending_requests(request):
    PendingApprovals = ApprovalModel.objects.filter(approvalStatus='pending')
    return HttpResponse(PendingApprovals)


def admin_mem_list(request):

    RetUserList = User_Model.objects.all()
    FinalRow = []
    for i in RetUserList:
        InternalRow = {}
        userDetails = senduserDet(i.user_UserName)
        print(userDetails)
        InternalRow['userName'] = i.user_Name
        InternalRow['summary_count'] = SummaryModel.objects.filter(
            summary_added_by=userDetails).count()

        InternalRow['total_approvals'] = ApprovalModel.objects.filter(
            approdeval_sent_by=userDetails).count()
        InternalRow['total_approvals_pending'] = ApprovalModel.objects.filter(
            approdeval_sent_by=userDetails, approvalStatus='pending').count()

        InternalRow['translations_count'] = TranslationsModel.objects.filter(
            transDoneBy=userDetails).count()
        InternalRow['contentCount'] = ContentModal.objects.filter(
            contentAddedBy=userDetails).count()
        FinalRow.append(InternalRow)
    context = {
        'Row': FinalRow
    }
    return render(request, "admin_pages/admin_mem_list.html", context)


def admin_approve_request(request, req_id):
    userDetSend = userDet(request.session['UserName'])
    ApproveRequest = ApprovalModel.objects.get(approvalID=req_id)
    if ApproveRequest.approvalType == 'trans':

        TranslationtoApproveRequest = TranslationsModel.objects.get(
            transID=ApproveRequest.approval_type_id)

        Var = TranslationsModel.objects.filter(
            transID=ApproveRequest.approval_type_id).update(
            transApprovalStatus=True, transApprovedBy=userDetSend)
        UpdateApprovalStatus = ApprovalModel.objects.filter(approvalID=req_id).update(
            approvalStatus='approved', approval_remaks="Approved Can publish to any sources", approvedBy=userDetSend)
        return JsonResponse({
            'approval': "OK",
            "Success_CODE": "Trans_APPROVED",
            'Request_ID': req_id,
            "Trans_ID": TranslationtoApproveRequest.transID
        })

    if ApproveRequest.approvalType == 'summary':
        return HttpResponse("Summary Detected")


def admin_reject_request(request, req_id):
    userDetSend = userDet(request.session['UserName'])
    ApproveRequest = ApprovalModel.objects.get(approvalID=req_id)
    if ApproveRequest.approvalType == 'trans':

        TranslationtoApproveRequest = TranslationsModel.objects.get(
            transID=ApproveRequest.approval_type_id)

        Var = TranslationsModel.objects.filter(
            transID=ApproveRequest.approval_type_id).update(
            transApprovalStatus=False, transApprovedBy=userDetSend)
        UpdateApprovalStatus = ApprovalModel.objects.filter(approvalID=req_id).update(
            approvalStatus='rejected', approval_remaks="Rejected Can not publish to any sources", approvedBy=userDetSend)
        return JsonResponse({
            'approval': "rejected",
            "Success_CODE": "Trans_REJECTED",
            'Request_ID': req_id,
            "Trans_ID": TranslationtoApproveRequest.transID
        })


def userDet(userName):
    userDetailsfromDB = AdminModel.objects.get(admin_usr_nm=userName)
    return userDetailsfromDB


def senduserDet(userName):
    userDetailsfromDB = User_Model.objects.get(user_UserName=userName)
    return userDetailsfromDB
